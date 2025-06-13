import os
import requests
from typing import List, Dict, Any
from langchain_core.tools import tool
from agent.model_factory import create_chat_model


@tool
def web_search_tool(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    使用搜索API进行网络搜索
    
    Args:
        query: 搜索查询
        num_results: 返回结果数量
        
    Returns:
        List[Dict]: 搜索结果列表
    """
    # 这里可以集成多种搜索API，比如：
    # 1. Google Custom Search API
    # 2. Bing Search API  
    # 3. DuckDuckGo API
    # 4. SerpAPI
    
    # 示例：使用DuckDuckGo搜索（免费）
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=num_results)
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                })
        return results
    except ImportError:
        # 如果没有安装duckduckgo_search，返回模拟结果
        return [
            {
                "title": f"搜索结果 {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"这是关于 '{query}' 的搜索结果 {i+1} 的摘要内容。"
            }
            for i in range(num_results)
        ]


def perform_web_search_with_qwen(
    query: str, 
    model_name: str = "qwen-turbo",
    num_results: int = 5
) -> Dict[str, Any]:
    """
    使用通义千问模型进行网络搜索和内容分析
    
    Args:
        query: 搜索查询
        model_name: 通义千问模型名称
        num_results: 搜索结果数量
        
    Returns:
        Dict: 包含搜索结果和分析内容的字典
    """
    # 1. 执行网络搜索
    search_results = web_search_tool.invoke({"query": query, "num_results": num_results})
    
    # 2. 使用通义千问分析搜索结果
    llm = create_chat_model(
        model_name=model_name,
        provider="qwen",
        temperature=0.1,
        max_retries=2,
    )
    
    # 构建分析提示
    search_content = "\n\n".join([
        f"标题: {result['title']}\n网址: {result['url']}\n摘要: {result['snippet']}"
        for result in search_results
    ])
    
    analysis_prompt = f"""
基于以下搜索结果，请提供一个综合性的分析和总结：

搜索查询: {query}

搜索结果:
{search_content}

请提供：
1. 对搜索结果的综合分析
2. 关键信息的提取和整理
3. 相关的引用和来源

请确保回答准确、全面，并包含适当的引用。
"""
    
    # 获取模型分析
    analysis_response = llm.invoke(analysis_prompt)
    
    # 构建返回结果
    return {
        "search_query": query,
        "search_results": search_results,
        "analysis": analysis_response.content,
        "sources_gathered": [
            {
                "title": result["title"],
                "url": result["url"],
                "short_url": result["url"],  # 简化处理
                "value": result["url"],
                "label": result["title"][:50] + "..." if len(result["title"]) > 50 else result["title"]
            }
            for result in search_results
        ]
    } 