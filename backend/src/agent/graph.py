import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
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
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# Global variable to store MCP tools
amap_tools = None

async def load_amap_mcp_tools():
    """加载高德MCP工具。"""
    global amap_tools
    
    if amap_tools is not None:
        return amap_tools
    
    try:
        # 配置高德MCP服务器参数
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@amap/amap-maps-mcp-server"],
            env={
                "AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY", "")
            }
        )
        
        # 建立与MCP服务器的会话并加载工具
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                amap_tools = await load_mcp_tools(session)
                
        return amap_tools
        
    except Exception as e:
        print(f"Failed to load Amap MCP tools: {e}")
        return []


def has_location_intent(query: str) -> bool:
    """检测查询是否包含地理位置意图。
    
    Args:
        query: 搜索查询字符串
        
    Returns:
        如果查询包含地理位置意图，返回True
    """
    location_indicators = [
        "附近", "周边", "周围", "旁边", "地址", "位置", "在哪",
        "怎么走", "路线", "导航", "距离", "坐标",
        "市", "区", "县", "街", "路", "广场", "商场", "中心"
    ]
    
    return any(indicator in query for indicator in location_indicators)


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to both web research and amap research nodes.

    This is used to spawn parallel research tasks for each query.
    """
    tasks = []
    
    for idx, search_query in enumerate(state["query_list"]):
        # 总是进行网络搜索
        tasks.append(Send("web_research", {"search_query": search_query, "id": int(idx)}))
        
        # 检测是否需要地理位置搜索
        if has_location_intent(search_query):
            tasks.append(Send("amap_research", {"search_query": search_query, "id": int(idx)}))
    
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
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
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


async def amap_research(state: AmapSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs location-based research using Amap MCP service.

    Executes location-based search using Amap (高德地图) MCP service.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable

    Returns:
        Dictionary with state update, including amap_research_result
    """
    try:
        search_query = state["search_query"]
        
        # 加载高德MCP工具
        tools = await load_amap_mcp_tools()
        
        if not tools:
            error_result = f"高德MCP工具加载失败\n查询：{search_query}"
            return {
                "search_query": [state["search_query"]],
                "amap_research_result": [error_result],
            }
        
        # 创建一个简单的LLM来使用MCP工具
        configurable = Configuration.from_runnable_config(config)
        llm = ChatGoogleGenerativeAI(
            model=configurable.query_generator_model,
            temperature=0.1,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        
        # 将LLM与MCP工具绑定
        llm_with_tools = llm.bind_tools(tools)
        
        # 创建专门的高德搜索提示
        amap_prompt = f"""请使用高德地图工具来搜索以下查询的相关信息：
        
查询：{search_query}

请按照以下步骤：
1. 如果查询包含地点名称，先使用地理编码工具获取该地点的坐标
2. 然后使用周边搜索工具查找相关的POI信息
3. 提供详细的搜索结果，包括地点名称、地址、距离等信息

请用中文回答，并提供结构化的结果。"""
        
        # 调用LLM处理查询
        response = await llm_with_tools.ainvoke(amap_prompt)
        
        # 格式化结果
        formatted_result = f"**高德地图搜索结果**\n查询：{search_query}\n\n{response.content}"
        
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
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reflection_model = state.get("reflection_model") or configurable.reflection_model

    # Format the prompt
    current_date = get_current_date()
    # 合并网络搜索和高德搜索结果
    all_research_results = state.get("web_research_result", []) + state.get("amap_research_result", [])
    
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(all_research_results),
    )
    # init Reflection Model
    llm = ChatGoogleGenerativeAI(
        model=reflection_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
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
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
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
        for idx, follow_up_query in enumerate(state["follow_up_queries"]):
            task_id = state["number_of_ran_queries"] + int(idx)
            
            # 总是进行网络搜索
            tasks.append(Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": task_id,
                }
            ))
            
            # 检测是否需要地理位置搜索
            if has_location_intent(follow_up_query):
                tasks.append(Send(
                    "amap_research",
                    {
                        "search_query": follow_up_query,
                        "id": task_id,
                    }
                ))
        
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

    # init Answer Model, default to Gemini 2.5 Pro
    llm = ChatGoogleGenerativeAI(
        model=answer_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
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


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("amap_research", amap_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in parallel branches
builder.add_conditional_edges(
    "generate_query", continue_to_research, ["web_research", "amap_research"]
)
# Reflect on both web research and amap research
builder.add_edge("web_research", "reflection")
builder.add_edge("amap_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "amap_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
