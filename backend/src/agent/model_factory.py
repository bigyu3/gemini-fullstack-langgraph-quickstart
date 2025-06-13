import os
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatTongyi
from langchain_core.language_models.chat_models import BaseChatModel
from agent.configuration import Configuration


def create_chat_model(
    model_name: str, 
    provider: str = "gemini", 
    temperature: float = 0.0, 
    max_retries: int = 2,
    **kwargs: Any
) -> BaseChatModel:
    """
    创建聊天模型实例
    
    Args:
        model_name: 模型名称
        provider: 模型提供商 ('gemini' 或 'qwen')
        temperature: 温度参数
        max_retries: 最大重试次数
        **kwargs: 其他参数
        
    Returns:
        BaseChatModel: 聊天模型实例
    """
    if provider.lower() == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            api_key=os.getenv("GEMINI_API_KEY"),
            **kwargs
        )
    elif provider.lower() == "qwen":
        # 通义千问模型映射
        qwen_model_mapping = {
            "gemini-2.0-flash": "qwen-turbo",
            "gemini-2.5-flash-preview-04-17": "qwen-plus", 
            "gemini-2.5-pro-preview-05-06": "qwen-max",
            # 直接使用通义千问模型名
            "qwen-turbo": "qwen-turbo",
            "qwen-plus": "qwen-plus", 
            "qwen-max": "qwen-max",
            "qwen-long": "qwen-long",
        }
        
        # 如果是Gemini模型名，映射到对应的通义千问模型
        actual_model = qwen_model_mapping.get(model_name, model_name)
        
        return ChatTongyi(
            model=actual_model,
            temperature=temperature,
            max_retries=max_retries,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            **kwargs
        )
    else:
        raise ValueError(f"不支持的模型提供商: {provider}")


def get_model_config_for_provider(provider: str) -> Dict[str, str]:
    """
    根据提供商获取默认模型配置
    
    Args:
        provider: 模型提供商
        
    Returns:
        Dict[str, str]: 包含各种用途的默认模型名称
    """
    if provider.lower() == "gemini":
        return {
            "query_generator_model": "gemini-2.0-flash",
            "reflection_model": "gemini-2.5-flash-preview-04-17", 
            "answer_model": "gemini-2.5-pro-preview-05-06"
        }
    elif provider.lower() == "qwen":
        return {
            "query_generator_model": "qwen-turbo",
            "reflection_model": "qwen-plus",
            "answer_model": "qwen-max"
        }
    else:
        raise ValueError(f"不支持的模型提供商: {provider}") 