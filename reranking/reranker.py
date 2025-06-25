import os

from langchain.retrievers import  ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors import CohereRerank


def get_reranker(
    retriever:  BaseRetriever,
    rerank_model: str = "rerank-english-v3.0",
    top_n: int = 10
) -> ContextualCompressionRetriever:
    if os.getenv("COHERE_API_KEY") is None:
        raise ValueError("请设置 COHERE_API_KEY 环境变量以使用 Cohere Rerank 模型。")

    compressor = CohereRerank(
        model=rerank_model,
        top_n=top_n,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    return ContextualCompressionRetriever(
        retriever=retriever,
        document_compressor=compressor
    )