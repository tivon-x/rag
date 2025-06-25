from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.llm import ChatModel
from utils.parser import line_list_output_parser


def get_rewritten_query_retriever(
    retriever: BaseRetriever,
    llm: ChatModel
    ) -> Runnable:
    """
    创建一个重写查询检索器，该检索器使用 LLM 重写用户输入的查询，并使用基础检索器检索相关文档。
    Args:
        retriever (BaseRetriever): 用于检索文档的基础检索器。
        llm (ChatModel): 用于重写查询的语言模型。
    Returns:
        Runnable: 一个重写查询检索器实例，通过invoke()方法可以执行查询重写和文档检索。
    """ 
    template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

        Original query: {original_query}

        Rewritten query:"""
    prompt_rewrite = ChatPromptTemplate.from_template(template)

    return prompt_rewrite | llm | retriever


def get_multi_query_retriever(retriever: BaseRetriever, llm: ChatModel) -> MultiQueryRetriever:
    """
    创建一个 MultiQueryRetriever 实例，该实例通过使用 LLM 从不同角度为给定的用户输入查询生成多个查询。
    
    对于每个查询，它检索一组相关文档，并取所有查询的唯一并集，以获得更大的潜在相关文档集。

    Args:
        retriever (BaseRetriever): 用于检索文档的基础检索器。
        llm (ChatModel): 用于生成查询的语言模型。
    Returns:
        MultiQueryRetriever: 一个多查询检索器实例。
    """
    return MultiQueryRetriever(retriever=retriever, llm=llm)


def get_rag_fusion_retriever(
    retriever: BaseRetriever,
    llm: ChatModel
) -> Runnable:
    """ 创建一个 RAG Fusion 检索器，该检索器使用 LLM 从用户输入查询生成多个查询，并使用 RRF（Reciprocal Rank Fusion）算法融合检索结果。
    
    Args:
        retriever (BaseRetriever): 用于检索文档的基础检索器。
        llm (ChatModel): 用于生成查询的语言模型。
    Returns:
        Runnable: 一个 RAG Fusion 检索器实例，通过invoke()方法可以执行查询生成和文档检索。
    
    
    """

    def reciprocal_rank_fusion(results: list[list[Document]], k=60, n=7) -> list[Document]:
        """
        接受多个排名文档列表、RRF公式中使用的可选参数k、返回的文档数量
        """

        # 初始化字典以保存每个唯一文档的融合分数
        fused_scores = {}

        # 遍历每个排名文档列表
        for docs in results:
            # 遍历列表中的每个文档及其排名（列表中的位置）
            for rank, doc in enumerate(docs):
                # 将文档 id 以用作key
                doc_id = doc.id
                # 如果文档尚未在fused_scores字典中，请将其初始分数添加为0
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0

                # 使用RRF公式更新文档的分数：1/（rank+k）
                fused_scores[doc_id] += 1 / (rank + k)

        # 根据fusion分数按降序对文档进行排序，以获得最终的重新排序结果
        reranked_results = [
            doc
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # 将重新排序的结果作为元组列表返回，每个元组包含文档
        return reranked_results[:n]
    
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = prompt_rag_fusion | llm | line_list_output_parser

    return generate_queries | retriever.map() | reciprocal_rank_fusion


def get_step_back_retriever(
    retriever: BaseRetriever,
    llm: ChatModel
) -> Runnable:
    """
    创建一个 Step Back 检索器，该检索器使用 LLM 从用户输入查询生成一个表示更抽象或更高层次的问题的回退查询，并使用基础检索器检索相关文档。

    这种方法强调通过提出一般性问题来理解更广泛的背景和基本概念，从而提供更大的视角。
    
    该方法包括使用例子来引导抽象问题的形成，并允许独立检索与原问题和后退问题相关的信息。
    
    这种双重检索过程可以提升理解并生成更全面的回答，特别适用于技术文档和教科书等需要大量概念知识的领域，通过分别处理高层次概念及其详细实现来增强实用性。
    
    Args:
        retriever (BaseRetriever): 用于检索文档的基础检索器。
        llm (ChatModel): 用于生成查询的语言模型。
    Returns:
        Runnable: 一个 Step Back 检索器实例，通过invoke()方法可以执行查询生成和文档检索。
    """
    def fusion_step_back(
        results: list[list[Document]]
    ) -> list[Document]:
        """
        接受多个文档列表，返回一个合并、去重后的文档列表。
        """
        # 将所有文档展平为一个列表
        all_docs = [doc for sublist in results for doc in sublist]
        # 使用 dict 去重
        unique_docs = {doc.id: doc for doc in all_docs}
        return list(unique_docs.values())


    template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
    Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

    Original query: {original_query}

    Step-back query:"""
    prompt_step_back = ChatPromptTemplate.from_template(template)

    return (
        {
            "step_back_query": prompt_step_back | llm | StrOutputParser(),
            "original_query": RunnablePassthrough(),
        }
        | RunnableLambda(lambda x: [x["original_query"], x["step_back_query"]])
        | retriever.map()
        | fusion_step_back
    )