# 通义千问模型配置指南

本项目现在支持阿里巴巴的通义千问模型。以下是配置和使用指南。

## 环境变量配置

在 `backend/` 目录下创建 `.env` 文件，添加以下配置：

```bash
# Gemini API Key (使用Gemini模型时需要)
GEMINI_API_KEY=your_gemini_api_key_here

# DashScope API Key (使用通义千问模型时需要)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# 模型提供商配置
# 选项: "gemini" 或 "qwen"
MODEL_PROVIDER=qwen

# 模型配置 (可选，不指定时使用默认值)
# 对于通义千问:
QUERY_GENERATOR_MODEL=qwen-turbo
REFLECTION_MODEL=qwen-plus
ANSWER_MODEL=qwen-max

# 研究配置
NUMBER_OF_INITIAL_QUERIES=3
MAX_RESEARCH_LOOPS=2
```

## 获取通义千问API密钥

1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 开通DashScope服务
4. 在API-KEY管理页面创建新的API密钥
5. 将API密钥复制到 `.env` 文件中的 `DASHSCOPE_API_KEY`

## 支持的通义千问模型

- `qwen-turbo`: 快速响应，适合查询生成
- `qwen-plus`: 平衡性能和质量，适合反思分析
- `qwen-max`: 最高质量，适合最终答案生成
- `qwen-long`: 长文本处理能力强

## 模型映射

当使用通义千问时，系统会自动将Gemini模型名映射到对应的通义千问模型：

- `gemini-2.0-flash` → `qwen-turbo`
- `gemini-2.5-flash-preview-04-17` → `qwen-plus`
- `gemini-2.5-pro-preview-05-06` → `qwen-max`

## 搜索功能

通义千问模型使用DuckDuckGo搜索引擎进行网络搜索，无需额外的搜索API密钥。

## 使用示例

```bash
# 设置使用通义千问
export MODEL_PROVIDER=qwen
export DASHSCOPE_API_KEY=your_api_key

# 启动服务
uv run langgraph dev
```

## 注意事项

1. 确保已安装所有依赖：`uv sync`
2. 通义千问模型的响应格式可能与Gemini略有不同
3. 搜索结果的引用格式会根据模型提供商自动调整
4. 建议在生产环境中使用 `qwen-max` 获得最佳质量 