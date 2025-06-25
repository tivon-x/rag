#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from langchain_huggingface import HuggingFaceEmbeddings


def get_hf_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   model_kwargs: dict = None) -> HuggingFaceEmbeddings:
    """
    获取 HuggingFace 的嵌入模型。

    Args:
        model_name (str): 模型名称，默认为 "sentence-transformers/all-MiniLM-L6-v2"。
        model_kwargs (dict): 模型的其他参数。

    Returns:
        HuggingFaceEmbeddings: 返回 HuggingFace 的嵌入模型实例。
    """
    if model_kwargs is None:
        model_kwargs = {}


    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
