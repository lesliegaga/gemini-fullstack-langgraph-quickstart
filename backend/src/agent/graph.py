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

# 尝试导入 tiktoken，如果失败则使用备用方案
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️ tiktoken 库未安装，将使用备用 token 计算方法")
    print("建议安装: pip install tiktoken")

# Qwen3模型配置常量
QWEN3_MAX_CONTEXT_LENGTH = 20000  # qwen3_32b模型的最大上下文长度
QWEN3_SAFE_MAX_TOKENS = 5000      # 安全的最大输出tokens，留充足空间给输入
QWEN3_SAFE_CONTEXT_LENGTH = 18000 # 安全的上下文检查长度，留缓冲空间

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
_mcp_init_lock = asyncio.Lock()  # 异步锁，防止并发初始化


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
    
    import os
    import threading
    
    current_pid = os.getpid()
    current_thread = threading.current_thread().ident
    
    # 双重检查锁定模式：先检查状态，避免不必要的锁等待
    if _mcp_initialized:
        # 检查连接健康状态
        if await check_mcp_health():
            print(f"🔄 MCP工具已初始化且健康，返回缓存的 {len(amap_tools)} 个工具 (PID: {current_pid})")
            return amap_tools
        else:
            print(f"⚠️ MCP连接不健康，尝试重新初始化... (PID: {current_pid})")
            _mcp_initialized = False
            _mcp_retry_count += 1
    
    print(f"🔒 尝试获取MCP初始化锁... (PID: {current_pid}, Thread: {current_thread})")
    
    # 使用异步锁防止并发初始化
    async with _mcp_init_lock:
        print(f"✅ 获得MCP初始化锁 (PID: {current_pid}, Thread: {current_thread})")
        
        # 再次检查状态，防止在等待锁的过程中其他线程已经完成初始化
        if _mcp_initialized:
            print(f"🔄 在锁内检查：MCP工具已初始化，返回缓存的 {len(amap_tools)} 个工具 (PID: {current_pid})")
            return amap_tools
        
        # 检查重试次数
        if _mcp_retry_count >= _max_mcp_retries:
            print(f"❌ MCP工具初始化失败次数过多（{_mcp_retry_count}次），停止重试 (PID: {current_pid})")
            amap_tools = []
            _mcp_initialized = True
            return amap_tools
        
        try:
            print(f"🚀 {'重新' if _mcp_initialized else '首次'}初始化MCP客户端... (尝试 {_mcp_retry_count + 1}/{_max_mcp_retries}, PID: {current_pid})")
            
            # 如果已有客户端，先关闭
            if mcp_client:
                try:
                    await mcp_client.aclose()
                except:
                    pass
            
            mcp_client = MultiServerMCPClient(MCP_SERVER_CONFIG)
            
            # 加载高德MCP工具
            amap_tools = await mcp_client.get_tools()
            _mcp_initialized = True
            _mcp_retry_count = 0  # 重置重试计数
            print(f"✅ 成功加载 {len(amap_tools)} 个高德MCP工具 (PID: {current_pid})")
            print(f"🎯 MCP初始化完成，状态标记为: {_mcp_initialized} (PID: {current_pid})")
            
        except Exception as e:
            print(f"⚠️ 高德MCP工具加载失败: {e} (PID: {current_pid})")
            print("系统将在没有高德地图支持的情况下运行")
            amap_tools = []
            _mcp_initialized = True  # 即使失败也标记为已初始化，避免重复尝试
            print(f"🎯 MCP初始化失败但状态已标记为: {_mcp_initialized} (PID: {current_pid})")
        
        return amap_tools


async def get_mcp_status():
    """获取MCP工具的当前状态（用于调试）"""
    global mcp_client, amap_tools, _mcp_initialized, _mcp_retry_count
    
    import os
    import threading
    
    current_pid = os.getpid()
    current_thread = threading.current_thread().ident
    
    status = {
        "pid": current_pid,
        "thread": current_thread,
        "initialized": _mcp_initialized,
        "retry_count": _mcp_retry_count,
        "tools_count": len(amap_tools) if amap_tools else 0,
        "client_exists": mcp_client is not None,
        "lock_locked": _mcp_init_lock.locked() if hasattr(_mcp_init_lock, 'locked') else "Unknown"
    }
    
    print(f"📊 MCP状态报告 (PID: {current_pid}):")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    return status


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
    
    def __init__(self, llm, tools, max_context_length=QWEN3_SAFE_CONTEXT_LENGTH):
        self.llm = llm
        self.tools = tools
        self.max_context_length = max_context_length
        self.tool_map = {tool.name: tool for tool in tools}
        self.total_prompt_tokens = 0  # 跟踪总的prompt tokens
        self.tokenizer = None  # 延迟初始化
        # 创建带工具的LLM链
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    async def _initialize_tokenizer_async(self):
        """异步初始化分词器"""
        if not TIKTOKEN_AVAILABLE:
            raise RuntimeError("tiktoken 库未安装，无法计算 token 数量")
        
        try:
            # 在线程池中执行阻塞操作
            tokenizer = await asyncio.to_thread(tiktoken.get_encoding, "cl100k_base")
            print("✅ tiktoken cl100k_base 编码器初始化成功")
            return tokenizer
        except Exception as e:
            print(f"⚠️ tiktoken cl100k_base 编码器初始化失败: {e}")
            try:
                # 备用编码器
                tokenizer = await asyncio.to_thread(tiktoken.get_encoding, "gpt2")
                print("✅ tiktoken gpt2 编码器初始化成功")
                return tokenizer
            except Exception as e2:
                print(f"⚠️ tiktoken gpt2 编码器初始化失败: {e2}")
                raise RuntimeError(f"tiktoken 编码器初始化失败: {e2}")
    
    async def _ensure_tokenizer(self):
        """确保分词器已初始化"""
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = await self._initialize_tokenizer_async()
        return self.tokenizer
    
    def get_total_prompt_tokens(self):
        """获取累积的prompt tokens总数"""
        return self.total_prompt_tokens
    
    async def _generate_final_response(self, messages, reason="达到限制"):
        """生成最终回答的通用方法，用于复用停止逻辑"""
        print(f"⚠️ {reason}，停止工具调用并生成最终回答")
        
        # 修改最后一条用户消息，添加停止指令
        stop_instruction = f"""您现在已经{reason}。您应该停止进行工具调用，并基于以上所有信息重新思考，提供您认为最可能的答案。请基于您迄今为止收集的所有信息提供一份全面的总结报告。"""
        
        # 添加停止指令作为新的用户消息
        messages.append(HumanMessage(content=stop_instruction))
        
        # 调用LLM生成最终回答 - 使用流式输出
        try:
            stream = self.llm.astream(messages, stream_usage=True)
            full = await anext(stream)
            async for chunk in stream:
                full += chunk
            final_response = full
        except Exception as e:
            print(f"⚠️ 流式输出失败，回退到普通调用: {e}")
            if 'full' in locals():
                print(f"📝 当前已保存的输出结果: {full}")
            final_response = await self.llm.ainvoke(messages)

        messages.append(final_response)
        return messages
    
    async def calculate_text_tokens(self, text):
        """使用分词工具计算文本的token数量"""
        if not text:
            return 0
        
        text_str = str(text)
        
        # 确保分词器已初始化
        tokenizer = await self._ensure_tokenizer()
        
        # 使用异步方式调用 tokenizer.encode
        tokens = await asyncio.to_thread(tokenizer.encode, text_str)
        return len(tokens)
            
    
    async def calculate_messages_tokens(self, messages):
        """使用分词工具计算消息列表的总 token 数量"""
        total_tokens = 0
        
        for message in messages:
            if hasattr(message, 'content') and message.content:
                # 计算消息内容的 token
                content_tokens = await self.calculate_text_tokens(message.content)
                total_tokens += content_tokens
            
            # 如果是工具调用消息，将 tool_calls 转换为字符串计算 token
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_str = str(message.tool_calls)
                tool_calls_tokens = await self.calculate_text_tokens(tool_calls_str)
                total_tokens += tool_calls_tokens
            else:
                raise ValueError(f"Unsupported message: {message}")
        
        return total_tokens
    
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
        
        # 初始化 token 计数
        self.total_prompt_tokens = await self.calculate_messages_tokens(messages)
        print(f"🚀 初始消息 token 数量: {self.total_prompt_tokens}")
        
        # 添加标记变量来追踪是否需要强制停止
        should_force_stop = False
        
        while iteration < max_iterations:
            iteration += 1
            
            self.total_prompt_tokens = await self.calculate_messages_tokens(messages)
            # 检查上下文长度（使用token数）
            current_tokens = self.get_total_prompt_tokens()
            print(f"🔄 迭代 {iteration}: 当前 token 数量: {current_tokens}")
            
            if current_tokens > self.max_context_length:
                messages = await self._generate_final_response(
                    messages, 
                    f"达到最大上下文长度限制: {current_tokens} > {self.max_context_length}"
                )
                break
            
            
            # 调用LLM获取下一步行动 - 使用流式输出
            try:
                stream = self.llm_with_tools.astream(messages, stream_usage=True)
                full = await anext(stream)
                async for chunk in stream:
                    full += chunk
                response = full
            except Exception as e:
                print(f"⚠️ 流式输出失败，回退到普通调用: {e}")
                if 'full' in locals():
                    print(f"📝 当前已保存的输出结果: {full}")
                print(f"📋 当前消息历史:")
                for i, msg in enumerate(messages):
                    role = getattr(msg, 'type', getattr(msg, 'role', 'unknown'))
                    msg_content = getattr(msg, 'content', str(msg))
                    
                    # 处理AIMessageChunk等没有实际content的消息
                    if not msg_content or msg_content.strip() == '':
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            content = f"[工具调用: {len(msg.tool_calls)}个工具]"
                        else:
                            content = "[空消息]"
                    elif len(str(msg_content)) > 100:
                        content = f"{str(msg_content)[:50]}...{str(msg_content)[-50:]}"
                    else:
                        content = str(msg_content)
                    print(f"  {i+1}. [{role}] {content}")
                response = await self.llm_with_tools.ainvoke(messages)
            messages.append(response)
            self.total_prompt_tokens = await self.calculate_messages_tokens(messages)

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
                                content=str(tool_result).replace("\n",""),
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            # 添加工具消息
                            messages.append(tool_message)
                        except Exception as e:
                            # 工具执行失败
                            error_message = f"Error executing tool {tool_name}: {str(e)}"
                            tool_message = ToolMessage(
                                content=error_message,
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            messages.append(tool_message)
                        
                        # 重新计算token并检查长度
                        self.total_prompt_tokens = await self.calculate_messages_tokens(messages)
                        if self.get_total_prompt_tokens() > self.max_context_length:
                            messages = await self._generate_final_response(
                                messages, 
                                f"工具执行后达到最大上下文长度限制: {self.get_total_prompt_tokens()} > {self.max_context_length}"
                            )
                            should_force_stop = True
                            break
            else:
                # 没有工具调用，结束循环
                break
            
            # 如果需要强制停止，跳出外层循环
            if should_force_stop:
                break
        
        # 检查是否因为达到最大迭代次数而退出循环
        if iteration >= max_iterations and not should_force_stop:
            # 检查最后一个响应是否还有工具调用，如果有则需要强制停止
            last_response = messages[-1] if messages else None
            if (last_response and hasattr(last_response, 'tool_calls') and 
                last_response.tool_calls):
                messages = await self._generate_final_response(
                    messages, 
                    f"达到最大迭代次数限制: {iteration}/{max_iterations}"
                )
        
        print(f"🎯 最终 token 数量: {self.get_total_prompt_tokens()}")
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
        max_tokens=QWEN3_SAFE_MAX_TOKENS,
        model="qwen3_32b",
        timeout=120,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.0
    )
    structured_llm = llm.with_structured_output(DualSearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = dual_query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the dual search queries using streaming
    try:
        stream = structured_llm.stream(formatted_prompt, stream_usage=True)
        full = next(stream)
        for chunk in stream:
            full += chunk
        result = full
    except Exception as e:
        print(f"⚠️ 流式输出失败，回退到普通调用: {e}")
        if 'full' in locals():
            print(f"📝 当前已保存的输出结果: {full}")
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
            max_tokens=QWEN3_SAFE_MAX_TOKENS,
            model="qwen3_32b",
            timeout=120,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.0
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
            agent = CustomReactAgent(llm, amap_tools, max_context_length=QWEN3_SAFE_CONTEXT_LENGTH)
            
            # 调用自定义React Agent处理查询并实际执行工具，设置递归限制为100
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": formatted_prompt}]},
                config={"recursion_limit": 30}
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
        max_tokens=QWEN3_SAFE_MAX_TOKENS,
        model="qwen3_32b",
        timeout=120,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.0
    )
    # Generate reflection using streaming
    structured_reflection_llm = llm.with_structured_output(DualReflection)
    try:
        stream = structured_reflection_llm.stream(formatted_prompt, stream_usage=True)
        full = next(stream)
        for chunk in stream:
            full += chunk
        result = full
    except Exception as e:
        print(f"⚠️ 流式输出失败，回退到普通调用: {e}")
        if 'full' in locals():
            print(f"📝 当前已保存的输出结果: {full}")
        result = structured_reflection_llm.invoke(formatted_prompt)

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
    
    # 过滤并平衡处理搜索结果
    def filter_valid_results(results):
        """过滤掉失败的搜索结果"""
        valid_results = []
        for result in results:
            # 过滤掉包含错误信息的结果
            if (result and 
                not any(error_keyword in result for error_keyword in [
                    "搜索失败", "Error code:", "返回无效响应", 
                    "搜索结果（无引用信息）", "Google Search API", 
                    "没有找到相关的搜索结果", "无法完成", "抱歉"
                ]) and
                len(result.strip()) > 50):  # 确保结果有实质内容
                valid_results.append(result.strip())
        return valid_results
    
    # 分别获取和过滤web搜索和高德搜索结果
    web_results = filter_valid_results(state.get("web_research_result", []))
    amap_results = filter_valid_results(state.get("amap_research_result", []))
    
    # 平衡合并结果，确保两种搜索结果得到平等对待
    balanced_results = []
    max_len = max(len(web_results), len(amap_results))
    
    for i in range(max_len):
        if i < len(web_results):
            balanced_results.append(f"**网络搜索发现：**\n{web_results[i]}")
        if i < len(amap_results):
            balanced_results.append(f"**地图位置信息：**\n{amap_results[i]}")
    
    # 如果没有有效结果，添加提示信息
    if not balanced_results:
        balanced_results = ["根据搜索结果，未能获取到详细的相关信息。建议您通过其他渠道获取更多信息。"]
    
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(balanced_results),
    )

    # init Answer Model, default to Qwen3 32B
    llm = ChatOpenAI(
        base_url="http://proxy2-search.proxy.amap.com/zjy_llm_qwen/v1",
        max_tokens=QWEN3_SAFE_MAX_TOKENS,
        model="qwen3_32b",
        timeout=120,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.0
    )
    # Generate final answer using streaming
    try:
        stream = llm.stream(formatted_prompt, stream_usage=True)
        full = next(stream)
        for chunk in stream:
            full += chunk
        result = full
    except Exception as e:
        print(f"⚠️ 流式输出失败，回退到普通调用: {e}")
        if 'full' in locals():
            print(f"📝 当前已保存的输出结果: {full}")
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
    import os
    current_pid = os.getpid()
    print(f"🏗️ 开始创建LangGraph图 (PID: {current_pid})")
    
    # 确保MCP工具已初始化（只初始化一次）
    print(f"🔧 检查MCP工具状态 (PID: {current_pid})")
    await get_mcp_status()
    
    await initialize_mcp_tools()
    
    print(f"✅ MCP工具初始化完成，开始创建图 (PID: {current_pid})")
    
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

    print(f"🎯 LangGraph图创建完成 (PID: {current_pid})")
    return builder.compile(
        name="pro-search-agent"
        # 注意：递归限制在新版本LangGraph中通过其他方式设置
        # 当前版本通过RunnableConfig在运行时传递recursion_limit
    )


# 为了向后兼容，保留同步图变量
# 但推荐使用make_graph()函数来获取图实例
graph = None
