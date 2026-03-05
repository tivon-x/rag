
import numpy as np
from langchain_core.prompts import PromptTemplate
from indexing.embeddings import get_embeddings


def cosine_similarity(X, Y):
    """计算余弦相似度"""
    X = np.array(X)
    Y = np.array(Y)
    # 计算点积
    dot_product = np.dot(X, Y.T)
    # 计算范数
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    # 计算余弦相似度
    return dot_product / (norm_X * norm_Y.T)


def semantic_routing(query: str, config: dict):
    embedding_config = config.get("embedding", {})
    embeddings = get_embeddings(embedding_config)
    
    prompt_templates: list[str] = config.get("prompt_templates", [])
    prompt_embeddings = embeddings.embed_documents(
        prompt_templates
    )

    query_embedding = embeddings.embed_query(query)

    # 计算余弦相似度
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]

    return PromptTemplate.from_template(most_similar)