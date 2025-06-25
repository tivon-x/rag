from typing import TypeAlias
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

ChatModel: TypeAlias = ChatOpenAI | ChatGroq

def get_llm(config: dict) -> ChatModel | None:
    provider = config.get("provider", "").lower()
    model = config.get("model", "").lower()
    api_key = config.get("api_key", None)
    api_base = config.get("api_base", None)
    model_config = config.get("model_config", {})

    if not provider or not model:
        raise ValueError("Provider and model must be specified in the config.")
    if not api_key:
        raise ValueError("API key must be provided in the config.")
    if not api_base:
        raise ValueError("API base must be provided in the config.")
    
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_base,
            **model_config
        )
    elif provider == "groq":
        return ChatGroq(
            model=model,
            groq_api_key=api_key,
            groq_api_base=api_base,
            **model_config
        )