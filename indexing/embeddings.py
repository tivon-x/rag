#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings


def get_hf_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder: str = "./hf-cache",
    model_kwargs: dict | None = None,
) -> HuggingFaceEmbeddings:
    """
    获取 HuggingFace 的嵌入模型。

    Args:
        model_name (str): 模型名称，默认为 "sentence-transformers/all-MiniLM-L6-v2"。
        cache_folder (str): 模型缓存目录，默认为 "./hf-cache"。
        model_kwargs (dict): 模型的其他参数。

    Returns:
        HuggingFaceEmbeddings: 返回 HuggingFace 的嵌入模型实例。
    """
    if model_kwargs is None:
        model_kwargs = {}

    return HuggingFaceEmbeddings(model_name=model_name, cache_folder=cache_folder, model_kwargs=model_kwargs)


def get_cloud_embeddings(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs,
) -> OpenAIEmbeddings:
    """
    获取云端 OpenAI 兼容的嵌入模型。

    Args:
        model (str): 模型名称，默认为 "text-embedding-3-small"。
        api_key (str): API 密钥。
        api_base (str): API 基础地址。
        **kwargs: 其他传递给 OpenAIEmbeddings 的参数。

    Returns:
        OpenAIEmbeddings: 返回 OpenAI 兼容的嵌入模型实例。
    """
    if not api_key:
        raise ValueError("API key must be provided for cloud embeddings.")
    if not api_base:
        raise ValueError("API base must be provided for cloud embeddings.")

    return OpenAIEmbeddings(model=model, api_key=api_key, base_url=api_base, **kwargs) # type: ignore


def get_embeddings(config: dict) -> Embeddings:
    """
    获取嵌入模型实例。优先使用云端 OpenAI 兼容模型，如果没有配置则使用 HuggingFace。

    Args:
        config (dict): 嵌入配置，包含以下可选字段:
            - type: 嵌入类型，"cloud" 或 "huggingface"，默认优先 cloud
            - model: 云端模型名称
            - api_key: 云端 API 密钥
            - api_base: 云端 API 基础地址
            - model_name: HuggingFace 模型名称
            - model_kwargs: HuggingFace 模型参数

    Returns:
        Embeddings: 嵌入模型实例。
    """
    embedding_config = config.get("embedding", {})

    # 优先使用云端嵌入
    api_key = (
        embedding_config.get("api_key")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    api_base = (
        embedding_config.get("api_base")
        or os.getenv("EMBEDDING_API_BASE")
        or os.getenv("OPENAI_API_BASE")
    )
    model = embedding_config.get(
        "model", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # 如果配置了 api_key 和 api_base，使用云端嵌入
    if api_key and api_base:
        return get_cloud_embeddings(
            model=model,
            api_key=api_key,
            api_base=api_base,
            # 支持额外参数
            dimensions=embedding_config.get("dimensions"),
            timeout=embedding_config.get("timeout"),
        )

    # 回退到 HuggingFace 嵌入
    model_name = embedding_config.get(
        "model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_kwargs = embedding_config.get("model_kwargs", {})

    return get_hf_embeddings(model_name=model_name, model_kwargs=model_kwargs)
