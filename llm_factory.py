import os
from openai import OpenAI

# 定义支持的模型后端配置
# 现在绝大多数国产大模型都兼容 OpenAI 的接口格式，只要换 base_url 和 api_key 即可
MODEL_CONFIGS = {
    "zhipu": {
        "api_key": os.getenv("ZHIPU_API_KEY"),
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model_name": "glm-4-flash"
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    },
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-turbo"
    }
}

def get_llm_client(provider="zhipu"):
    """
    模型工厂函数：根据传入的 provider 返回对应的 Client 和 Model Name。
    业务代码不需要关心底层是哪家模型，只需要调用这个函数。
    """
    config = MODEL_CONFIGS.get(provider.lower())
    if not config:
        raise ValueError(f"不支持的模型提供商: {provider}")
    
    if not config["api_key"]:
        raise ValueError(f"环境变量中未找到 {provider} 的 API Key，请检查配置。")
    
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    return client, config["model_name"]

def call_llm(prompt, provider="zhipu", temperature=0.1):
    """
    统一的调用入口，封装了异常处理和 Token 统计。
    """
    client, model_name = get_llm_client(provider)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        content = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
        return content, tokens_used
    except Exception as e:
        print(f"❌ 调用 {provider} 模型失败: {e}")
        raise e