
from langchain.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from indexing.embeddings import get_hf_embeddings


def semantic_routing(query: str, config: dict):
    embedding_config = config.get("embedding", {})
    embeddings = get_hf_embeddings(embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                                       embedding_config.get("model_kwargs", {}))
    
    prompt_templates: list[str] = config.get("prompt_templates", [])
    prompt_embeddings = embeddings.embed_documents(
        prompt_templates
    )

    query_embedding = embeddings.embed_query(query)

    # 计算余弦相似度
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]

    return PromptTemplate.from_template(most_similar)