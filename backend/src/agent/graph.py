import os

from agent.tools_and_schemas import SearchQueryList, Reflection, DualSearchQueryList, DualReflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
    AmapSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    dual_query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    dual_reflection_instructions,
    answer_instructions,
    amap_searcher_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# MCP配置常量
MCP_HEALTH_CHECK_INTERVAL = 300  # 健康检查间隔（秒）
MCP_MAX_RETRIES = 3  # 最大重试次数
MCP_SERVER_CONFIG = {
    "amap": {
        "command": "npx",
        "args": ["-y", "@amap/amap-maps-mcp-server"],
        "transport": "stdio",
        "env": {
            "AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY", "")
        }
    }
}

# MCP客户端和工具存储
mcp_client = None
amap_tools = None
_mcp_initialized = False  # 标记是否已初始化
_mcp_last_health_check = 0  # 最后健康检查时间
_mcp_health_check_interval = MCP_HEALTH_CHECK_INTERVAL  # 健康检查间隔（秒）
_mcp_retry_count = 0  # 重试次数
_max_mcp_retries = MCP_MAX_RETRIES  # 最大重试次数


async def check_mcp_health():
    """检查MCP连接的健康状态"""
    global mcp_client, amap_tools, _mcp_last_health_check
    
    import time
    current_time = time.time()
    
    # 如果距离上次检查时间太短，跳过检查
    if current_time - _mcp_last_health_check < _mcp_health_check_interval:
        return amap_tools is not None and len(amap_tools) > 0
    
    _mcp_last_health_check = current_time
    
    try:
        if mcp_client and amap_tools:
            # 尝试获取工具列表来验证连接是否正常
            tools = await mcp_client.get_tools()
            if tools and len(tools) > 0:
                print(f"✅ MCP连接健康检查通过，当前有 {len(tools)} 个工具")
                return True
            else:
                print("⚠️ MCP连接健康检查失败：工具列表为空")
                return False
        else:
            print("⚠️ MCP连接健康检查失败：客户端或工具未初始化")
            return False
    except Exception as e:
        print(f"⚠️ MCP连接健康检查异常：{e}")
        return False


async def initialize_mcp_tools():
    """初始化MCP工具（单例模式，只初始化一次）"""
    global mcp_client, amap_tools, _mcp_initialized, _mcp_retry_count
    
    if _mcp_initialized:
        # 检查连接健康状态
        if await check_mcp_health():
            print(f"🔄 MCP工具已初始化且健康，返回缓存的 {len(amap_tools)} 个工具")
            return amap_tools
        else:
            print("⚠️ MCP连接不健康，尝试重新初始化...")
            _mcp_initialized = False
            _mcp_retry_count += 1
    
    # 检查重试次数
    if _mcp_retry_count >= _max_mcp_retries:
        print(f"❌ MCP工具初始化失败次数过多（{_mcp_retry_count}次），停止重试")
        amap_tools = []
        _mcp_initialized = True
        return amap_tools
    
    try:
        print(f"🚀 {'重新' if _mcp_initialized else '首次'}初始化MCP客户端... (尝试 {_mcp_retry_count + 1}/{_max_mcp_retries})")
        
        # 如果已有客户端，先关闭
        if mcp_client:
            try:
                await mcp_client.aclose()
            except:
                pass
        
        mcp_client = MultiServerMCPClient(
            MCP_SERVER_CONFIG
        )
        
        # 加载高德MCP工具
        amap_tools = await mcp_client.get_tools()
        _mcp_initialized = True
        _mcp_retry_count = 0  # 重置重试计数
        print(f"✅ 成功加载 {len(amap_tools)} 个高德MCP工具")
        
    except Exception as e:
        print(f"⚠️ 高德MCP工具加载失败: {e}")
        print("系统将在没有高德地图支持的情况下运行")
        amap_tools = []
        _mcp_initialized = True  # 即使失败也标记为已初始化，避免重复尝试
    
    return amap_tools


async def get_amap_tools():
    """获取高德MCP工具（延迟加载 + 健康检查）"""
    if not _mcp_initialized:
        return await initialize_mcp_tools()
    
    # 检查连接健康状态
    if await check_mcp_health():
        return amap_tools
    else:
        print("🔄 MCP连接不健康，重新初始化...")
        return await initialize_mcp_tools()


class CustomReactAgent:
    """自定义React Agent，支持上下文长度检查和智能停止机制"""
    
    def __init__(self, llm, tools, max_context_length=15000):
        self.llm = llm
        self.tools = tools
        self.max_context_length = max_context_length
        self.tool_map = {tool.name: tool for tool in tools}
        self.total_prompt_tokens = 0  # 跟踪总的prompt tokens
    
    def get_total_prompt_tokens(self):
        """获取累积的prompt tokens总数"""
        return self.total_prompt_tokens
    
    def calculate_text_tokens(self, text):
        """计算文本的token数量"""
        try:
            # 对于 ChatOpenAI，使用 get_num_tokens 方法
            if hasattr(self.llm, 'get_num_tokens'):
                return self.llm.get_num_tokens(text)
            # 对于其他 LLM，使用简单的字符估算（粗略估算）
            else:
                # 粗略估算：英文约4个字符1个token，中文约2个字符1个token
                english_chars = sum(1 for c in text if ord(c) < 128)
                chinese_chars = len(text) - english_chars
                return english_chars // 4 + chinese_chars // 2
        except Exception:
            # 如果计算失败，使用字符数作为后备方案
            return len(text) // 3
    
    def update_token_usage(self, response):
        """从LLM响应中提取并更新token使用情况"""
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'total_tokens'):
                self.total_prompt_tokens = response.usage.total_tokens
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            # 尝试从response_metadata中获取token信息
            metadata = response.response_metadata
            if 'token_usage' in metadata:
                token_usage = metadata['token_usage']
                if 'total_tokens' in token_usage:
                    self.total_prompt_tokens = token_usage['total_tokens']
        print(f"更新token使用情况: {self.total_prompt_tokens} usage: {response.usage if hasattr(response, 'usage') else 'None'}")
    
    def add_tool_message_tokens(self, tool_message):
        """计算并添加ToolMessage内容的token数量到总计数中"""
        if hasattr(tool_message, 'content') and tool_message.content:
            content_tokens = self.calculate_text_tokens(str(tool_message.content))
            self.total_prompt_tokens += content_tokens
            print(f"ToolMessage 内容添加了 {content_tokens} 个tokens，总计: {self.total_prompt_tokens}")
    
    async def ainvoke(self, input_data, config=None):
        """异步执行React Agent逻辑，支持上下文长度检查"""
        messages = input_data.get("messages", [])
        max_iterations = config.get("recursion_limit", 100) if config else 100
        
        # 确保消息格式正确
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    formatted_messages.append(AIMessage(content=msg["content"]))
                else:
                    formatted_messages.append(HumanMessage(content=str(msg)))
            else:
                formatted_messages.append(msg)
        
        messages = formatted_messages
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 检查上下文长度（使用token数）
            current_tokens = self.get_total_prompt_tokens()
            
            if current_tokens > self.max_context_length:
                # 修改最后一条用户消息，添加停止指令
                stop_instruction = """您现在已经达到了可以处理的最大上下文长度。您应该停止进行工具调用，并基于以上所有信息重新思考，提供您认为最可能的答案。请基于您迄今为止收集的所有信息提供一份全面的总结报告。"""
                
                # 添加停止指令作为新的用户消息
                messages.append(HumanMessage(content=stop_instruction))
                
                # 调用LLM生成最终回答
                final_response = await self.llm.ainvoke(messages)
                # 更新token使用情况
                self.update_token_usage(final_response)
                messages.append(final_response)
                break
            
            # 创建带工具的LLM链
            llm_with_tools = self.llm.bind_tools(self.tools)
            
            # 调用LLM获取下一步行动
            response = await llm_with_tools.ainvoke(messages)
            # 更新token使用情况
            self.update_token_usage(response)
            messages.append(response)

            if self.get_total_prompt_tokens() > self.max_context_length:
                print(f"当前token数量: {self.get_total_prompt_tokens()} 超过最大上下文长度: {self.max_context_length}")
                continue
            
            # 检查是否需要调用工具
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # 处理工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call.get("id", "")
                    
                    if tool_name in self.tool_map:
                        try:
                            # 执行工具
                            tool_result = await self.tool_map[tool_name].ainvoke(tool_args)
                            # 创建工具消息
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            # 添加工具消息
                            messages.append(tool_message)
                            # 计算并添加ToolMessage内容的token数量
                            self.add_tool_message_tokens(tool_message)
                        except Exception as e:
                            # 工具执行失败
                            error_message = f"Error executing tool {tool_name}: {str(e)}"
                            tool_message = ToolMessage(
                                content=error_message,
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            messages.append(tool_message)
                            # 计算并添加错误消息的token数量
                            self.add_tool_message_tokens(tool_message)
                            if self.get_total_prompt_tokens() > self.max_context_length:
                                print(f"当前token数量: {self.get_total_prompt_tokens()} 超过最大上下文长度: {self.max_context_length}")
                                break
            else:
                # 没有工具调用，结束循环
                break
        
        return {"messages": messages}

# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates dual search queries based on the User's question.

    Uses Qwen3 32B to create optimized search queries for both web research and map search
    based on the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including web_query_list and map_query_list
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Qwen3 32B
    llm = ChatOpenAI(
        base_url="http://proxy2-search.proxy.amap.com/zjy_llm_qwen/v1",
        max_tokens=10000,
        model="qwen3_32b",
        timeout=120,
        temperature=1.0,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(DualSearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = dual_query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the dual search queries
    result = structured_llm.invoke(formatted_prompt)
    
    # Convert to Query format for consistency
    web_queries = [{"query": q, "rationale": result.web_rationale} for q in result.web_queries]
    map_queries = [{"query": q, "rationale": result.map_rationale} for q in result.map_queries]
    
    return {
        "web_query_list": web_queries,
        "map_query_list": map_queries
    }


def continue_to_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to both web research and amap research nodes.

    This is used to spawn parallel research tasks for each query.
    """
    tasks = []
    task_id = 0
    
    # 处理网络搜索查询
    for web_query in state["web_query_list"]:
        tasks.append(Send("web_research", {"search_query": web_query["query"], "id": int(task_id)}))
        task_id += 1
    
    # 处理地图搜索查询
    for map_query in state["map_query_list"]:
        tasks.append(Send("amap_research", {"search_query": map_query["query"], "id": int(task_id)}))
        task_id += 1
    
    return tasks


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    
    # 检查响应是否有效
    if not response or not response.candidates:
        error_result = f"Google Search API 返回无效响应\n查询：{state['search_query']}"
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [error_result],
        }
    
    candidate = response.candidates[0]
    
    # 检查是否有 grounding_metadata 和 grounding_chunks
    if (not hasattr(candidate, "grounding_metadata") or 
        not candidate.grounding_metadata or 
        not hasattr(candidate.grounding_metadata, "grounding_chunks") or
        not candidate.grounding_metadata.grounding_chunks):
        
        # 如果没有 grounding 信息，返回原始响应文本
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [f"搜索结果（无引用信息）：\n{response.text}"],
        }
    
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        candidate.grounding_metadata.grounding_chunks, state["id"]
    )
    
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def amap_research(state: AmapSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs location-based research using Amap MCP service.

    Executes location-based search using Amap (高德地图) MCP service with React agent.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable

    Returns:
        Dictionary with state update, including amap_research_result
    """
    try:
        search_query = state["search_query"]
        
        # 检查预加载的工具
        if not amap_tools:
            error_result = f"高德MCP工具未初始化\n查询：{search_query}"
            return {
                "search_query": [state["search_query"]],
                "amap_research_result": [error_result],
            }
        
        # 创建LLM来使用MCP工具
        configurable = Configuration.from_runnable_config(config)
        llm = ChatOpenAI(
            base_url="http://proxy2-search.proxy.amap.com/zjy_llm_qwen/v1",
            max_tokens=15000,
            model="qwen3_32b",
            timeout=120,
            temperature=0.1,
            max_retries=2,
        )
        
        # 使用优化的高德搜索提示
        current_date = get_current_date()
        formatted_prompt = amap_searcher_instructions.format(
            search_query=search_query,
            current_date=current_date
        )
        
        # 定义异步执行函数
        async def execute_amap_research():
            # 创建自定义React Agent来执行工具调用，支持上下文长度检查
            agent = CustomReactAgent(llm, amap_tools, max_context_length=15000)
            
            # 调用自定义React Agent处理查询并实际执行工具，设置递归限制为100
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": formatted_prompt}]},
                config={"recursion_limit": 100}
            )
            
            # 获取最后一条消息的内容
            last_message = response["messages"][-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # 在同步函数中运行异步代码
        result_content = asyncio.run(execute_amap_research())
        
        # 格式化结果
        formatted_result = f"**高德地图搜索结果**\n查询：{search_query}\n\n{result_content}"
        
        return {
            "search_query": [state["search_query"]],
            "amap_research_result": [formatted_result],
        }
        
    except Exception as e:
        # 如果高德搜索失败，返回错误信息
        error_result = f"高德地图搜索失败：{str(e)}\n查询：{state['search_query']}"
        return {
            "search_query": [state["search_query"]],
            "amap_research_result": [error_result],
        }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries for both web and map search. Uses structured output to extract
    the follow-up queries in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including dual follow-up queries
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reflection_model = state.get("reflection_model") or configurable.reflection_model

    # Format the prompt
    current_date = get_current_date()
    # 合并网络搜索和高德搜索结果
    all_research_results = state.get("web_research_result", []) + state.get("amap_research_result", [])
    
    formatted_prompt = dual_reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(all_research_results),
    )
    # init Reflection Model
    llm = ChatOpenAI(
        base_url="http://proxy2-search.proxy.amap.com/zjy_llm_qwen/v1",
        max_tokens=15000,
        model="qwen3_32b",
        timeout=120,
        temperature=1.0,
        max_retries=2,
    )
    result = llm.with_structured_output(DualReflection).invoke(formatted_prompt)

    # Combine web and map follow-up queries for backward compatibility
    combined_follow_up_queries = result.web_follow_up_queries + result.map_follow_up_queries

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": combined_follow_up_queries,
        "web_follow_up_queries": result.web_follow_up_queries,
        "map_follow_up_queries": result.map_follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit or task list for follow-up research
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        tasks = []
        task_id = state["number_of_ran_queries"]
        
        # 处理网络搜索后续查询
        for web_follow_up_query in state.get("web_follow_up_queries", []):
            tasks.append(Send(
                "web_research",
                {
                    "search_query": web_follow_up_query,
                    "id": task_id,
                }
            ))
            task_id += 1
            
        # 处理地图搜索后续查询
        for map_follow_up_query in state.get("map_follow_up_queries", []):
            tasks.append(Send(
                "amap_research",
                {
                    "search_query": map_follow_up_query,
                    "id": task_id,
                }
            ))
            task_id += 1
        
        return tasks


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    answer_model = state.get("answer_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    # 合并网络搜索和高德搜索结果
    all_research_results = state.get("web_research_result", []) + state.get("amap_research_result", [])
    
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(all_research_results),
    )

    # init Answer Model, default to Qwen3 32B
    llm = ChatOpenAI(
        base_url="http://proxy2-search.proxy.amap.com/zjy_llm_qwen/v1",
        max_tokens=15000,
        model="qwen3_32b",
        timeout=120,
        temperature=0,
        max_retries=2,
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


async def make_graph():
    """创建并初始化LangGraph图，包括MCP工具的异步加载。
    
    参考langchain-mcp-adapters的"Using with LangGraph StateGraph"示例。
    
    Returns:
        编译好的LangGraph图实例
    """
    # 确保MCP工具已初始化（只初始化一次）
    await initialize_mcp_tools()
    
    # 创建Agent图
    builder = StateGraph(OverallState, config_schema=Configuration)

    # 定义节点
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("amap_research", amap_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)

    # 设置入口点
    builder.add_edge(START, "generate_query")
    
    # 添加条件边以继续并行分支的搜索查询
    builder.add_conditional_edges(
        "generate_query", continue_to_research, ["web_research", "amap_research"]
    )
    
    # 反思网络搜索和高德搜索
    builder.add_edge("web_research", "reflection")
    builder.add_edge("amap_research", "reflection")
    
    # 评估研究
    builder.add_conditional_edges(
        "reflection", evaluate_research, ["web_research", "amap_research", "finalize_answer"]
    )
    
    # 完成答案
    builder.add_edge("finalize_answer", END)

    return builder.compile(
        name="pro-search-agent"
        # 注意：递归限制在新版本LangGraph中通过其他方式设置
        # 当前版本通过RunnableConfig在运行时传递recursion_limit
    )


# 为了向后兼容，保留同步图变量
# 但推荐使用make_graph()函数来获取图实例
graph = None
