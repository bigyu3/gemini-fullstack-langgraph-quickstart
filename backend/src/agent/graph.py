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
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.model_factory import create_chat_model
from agent.search_tools import perform_web_search_with_qwen
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

# 检查至少有一个API密钥被设置
gemini_key = os.getenv("GEMINI_API_KEY")
dashscope_key = os.getenv("DASHSCOPE_API_KEY")

if gemini_key is None and dashscope_key is None:
    raise ValueError("至少需要设置 GEMINI_API_KEY 或 DASHSCOPE_API_KEY 中的一个")

# Used for Google Search API (only initialize if Gemini key is available)
genai_client = None
if gemini_key:
    genai_client = Client(api_key=gemini_key)


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

    # 检查用户输入是否是简单的问候或无意义的输入
    user_input = get_research_topic(state["messages"]).strip().lower()
    simple_greetings = ["hi", "hello", "hey", "你好", "哈喽", "嗨", "test", "测试"]
    
    if user_input in simple_greetings or len(user_input) < 3:
        # 对于简单问候，返回友好的回应查询
        return {"query_list": [f"如何友好地回应 '{user_input}' 这样的问候"]}

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init LLM based on provider
    llm = create_chat_model(
        model_name=configurable.query_generator_model,
        provider=configurable.model_provider,
        temperature=1.0,
        max_retries=2,
    )

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    # 根据模型提供商使用不同的方法
    if configurable.model_provider.lower() == "gemini":
        # Gemini支持结构化输出
        structured_llm = llm.with_structured_output(SearchQueryList)
        result = structured_llm.invoke(formatted_prompt)
        
        # 检查结果是否有效
        if result is None:
            raise ValueError("Gemini模型返回了空结果，请检查API密钥和模型配置")
        
        if not hasattr(result, 'query') or result.query is None:
            raise ValueError("Gemini模型返回的结果格式不正确，缺少query字段")
        
        return {"query_list": result.query}
    
    elif configurable.model_provider.lower() == "qwen":
        # 通义千问使用JSON格式提示
        json_prompt = formatted_prompt + f"""

请以JSON格式返回结果，格式如下：
{{
    "query": ["查询1", "查询2", "查询3"],
    "rationale": "选择这些查询的原因"
}}

请确保返回有效的JSON格式，包含{state["initial_search_query_count"]}个搜索查询。
"""
        
        response = llm.invoke(json_prompt)
        
        # 解析JSON响应
        import json
        try:
            # 尝试从响应中提取JSON
            content = response.content
            # 查找JSON部分
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                result_data = json.loads(json_str)
                
                if "query" in result_data and isinstance(result_data["query"], list):
                    return {"query_list": result_data["query"]}
                else:
                    raise ValueError("通义千问返回的JSON格式不正确，缺少query字段")
            else:
                raise ValueError("无法从通义千问响应中找到JSON格式")
                
        except json.JSONDecodeError as e:
            # 如果JSON解析失败，尝试简单的文本解析
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response.content}")
            
            # 生成默认查询
            topic = get_research_topic(state["messages"])
            default_queries = [
                f"{topic}",
                f"{topic} 最新信息",
                f"{topic} 详细介绍"
            ][:state["initial_search_query_count"]]
            
            return {"query_list": default_queries}
    
    else:
        raise ValueError(f"不支持的模型提供商: {configurable.model_provider}")


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using different search methods based on model provider.

    Executes a web search using either Google Search API (for Gemini) or alternative search methods (for Qwen).

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    
    if configurable.model_provider.lower() == "gemini":
        # 检查是否是简单问候的查询
        if "如何友好地回应" in state["search_query"] and "这样的问候" in state["search_query"]:
            # 对于简单问候，直接返回友好回应
            return {
                "sources_gathered": [],
                "search_query": [state["search_query"]],
                "web_research_result": ["你好！我是一个AI研究助手，很高兴见到你！我可以帮你研究各种问题，比如最新的科技趋势、历史事件、学术资料等。请告诉我你想了解什么，我会为你进行深入的研究和分析。"],
            }
        
        # 使用原有的Google Search API方法
        if genai_client is None:
            raise ValueError("使用 Gemini 模型需要设置 GEMINI_API_KEY")
            
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
    
    elif configurable.model_provider.lower() == "qwen":
        # 检查是否是简单问候的查询
        if "如何友好地回应" in state["search_query"] and "这样的问候" in state["search_query"]:
            # 对于简单问候，直接返回友好回应
            return {
                "sources_gathered": [],
                "search_query": [state["search_query"]],
                "web_research_result": ["你好！我是一个AI研究助手，很高兴见到你！我可以帮你研究各种问题，比如最新的科技趋势、历史事件、学术资料等。请告诉我你想了解什么，我会为你进行深入的研究和分析。"],
            }
        
        # 使用通义千问的搜索方法
        search_result = perform_web_search_with_qwen(
            query=state["search_query"],
            model_name=configurable.query_generator_model,
            num_results=5
        )
        
        return {
            "sources_gathered": search_result["sources_gathered"],
            "search_query": [state["search_query"]],
            "web_research_result": [search_result["analysis"]],
        }
    
    else:
        raise ValueError(f"不支持的模型提供商: {configurable.model_provider}")


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
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = create_chat_model(
        model_name=reasoning_model,
        provider=configurable.model_provider,
        temperature=1.0,
        max_retries=2,
    )
    
    # 根据模型提供商使用不同的方法
    if configurable.model_provider.lower() == "gemini":
        # Gemini支持结构化输出
        result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
        
        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_queries": result.follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }
    
    elif configurable.model_provider.lower() == "qwen":
        # 通义千问使用JSON格式提示
        json_prompt = formatted_prompt + """

请以JSON格式返回结果，格式如下：
{
    "is_sufficient": true/false,
    "knowledge_gap": "描述缺失的信息",
    "follow_up_queries": ["后续查询1", "后续查询2"]
}

请确保返回有效的JSON格式。
"""
        
        response = llm.invoke(json_prompt)
        
        # 解析JSON响应
        import json
        try:
            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                result_data = json.loads(json_str)
                
                return {
                    "is_sufficient": result_data.get("is_sufficient", False),
                    "knowledge_gap": result_data.get("knowledge_gap", "需要更多信息"),
                    "follow_up_queries": result_data.get("follow_up_queries", []),
                    "research_loop_count": state["research_loop_count"],
                    "number_of_ran_queries": len(state["search_query"]),
                }
            else:
                # 默认返回值
                return {
                    "is_sufficient": False,
                    "knowledge_gap": "需要更多信息",
                    "follow_up_queries": [get_research_topic(state["messages"]) + " 更多详情"],
                    "research_loop_count": state["research_loop_count"],
                    "number_of_ran_queries": len(state["search_query"]),
                }
                
        except json.JSONDecodeError as e:
            print(f"Reflection JSON解析失败: {e}")
            print(f"原始响应: {response.content}")
            
            # 默认返回值
            return {
                "is_sufficient": False,
                "knowledge_gap": "需要更多信息",
                "follow_up_queries": [get_research_topic(state["messages"]) + " 更多详情"],
                "research_loop_count": state["research_loop_count"],
                "number_of_ran_queries": len(state["search_query"]),
            }
    
    else:
        raise ValueError(f"不支持的模型提供商: {configurable.model_provider}")


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
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


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
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model
    llm = create_chat_model(
        model_name=reasoning_model,
        provider=configurable.model_provider,
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


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
