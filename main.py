import gradio as gr
import os
import json
from typing import Optional, Tuple, List, Dict, Any
import warnings

from dotenv import load_dotenv

# 导入各个模块
from indexing.indexer import Indexer
from indexing.verctorstore import VectorStore
from llm.llm import get_llm, ChatModel
from retrieval.retriever import get_similarity_retriever, BM25Retriever, FusionRetriever
from reranking.reranker import get_reranker
from query.query_translation import (
    get_rewritten_query_retriever,
    get_multi_query_retriever,
    get_rag_fusion_retriever,
    get_step_back_retriever,
)
from routing.routing import semantic_routing
from utils.logging import setup_logging
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

warnings.filterwarnings("ignore")

# 配置日志
logger = setup_logging(log_level="INFO")


class RAGApplication:
    def __init__(self):
        self.vectorstore: Optional[VectorStore] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.llm: Optional[ChatModel] = None
        self.retriever = None
        self.config = self._load_default_config()
        self.indexed_files: List[str] = []

    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "llm": {
                "model": os.getenv("LLM_MODEL"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_base": os.getenv("OPENAI_API_BASE"),
                "model_config": {"temperature": 0.7, "max_tokens": 1000},
            },
            "embedding": {
                # 云端嵌入配置（优先使用）
                "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                "api_key": os.getenv("EMBEDDING_API_KEY"),
                "api_base": os.getenv("EMBEDDING_API_BASE"),
                # HuggingFace 嵌入配置（回退使用）
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_kwargs": {},
            },
            "chunker": {
                "type": "recursive",
                "params": {"chunk_size": 512, "chunk_overlap": 64},
            },
            "vectorstore": {"persist_directory": "./vector_store"},
            "retrieval": {
                "type": "similarity",
                "k": 5,
                "use_reranker": False,
                "rerank_model": "rerank-english-v3.0",
                "rerank_top_n": 5,
            },
            "query_enhancement": {
                "type": "none",  # none, rewrite, multi_query, rag_fusion, step_back
            },
            "routing": {"enabled": False, "prompt_templates": []},
        }

    def update_config(self, config_json: str) -> str:
        """更新配置"""
        try:
            new_config = json.loads(config_json)
            self.config.update(new_config)
            logger.info("配置更新成功")
            return "✅ 配置更新成功"
        except json.JSONDecodeError as e:
            error_msg = f"❌ 配置JSON格式错误: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"❌ 配置更新失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def init_llm(self) -> str:
        """初始化LLM"""
        try:
            self.llm = get_llm(self.config["llm"])
            logger.info(f"LLM初始化成功: {self.config['llm']['model']}")
            return f"✅ LLM初始化成功: {self.config['llm']['model']}"
        except Exception as e:
            error_msg = f"❌ LLM初始化失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def index_documents(self, file_paths: List[str]) -> str:
        """索引文档"""
        try:
            if not file_paths:
                return "❌ 请选择要索引的文件"

            indexer = Indexer(self.config)

            for file_path in file_paths:
                if file_path in self.indexed_files:
                    continue

                result = indexer.index(file_path)
                if result is None:
                    continue

                vectorstore, bm25_index = result

                if self.vectorstore is None:
                    self.vectorstore = vectorstore
                    self.bm25_index = bm25_index
                else:
                    # 合并索引（这里简化处理，实际可能需要更复杂的合并逻辑）
                    pass

                self.indexed_files.append(file_path)
                logger.info(f"文档索引完成: {file_path}")

            return f"✅ 文档索引完成，已索引 {len(self.indexed_files)} 个文件"

        except Exception as e:
            error_msg = f"❌ 文档索引失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _get_retriever(self):
        """根据配置获取检索器"""
        if self.vectorstore is None or self.bm25_index is None:
            raise ValueError("请先索引文档")

        retrieval_config = self.config["retrieval"]
        retrieval_type = retrieval_config["type"]
        k = retrieval_config["k"]

        if retrieval_type == "similarity":
            base_retriever = get_similarity_retriever(self.vectorstore, k)
        elif retrieval_type == "bm25":
            base_retriever = BM25Retriever(self.bm25_index, k)
        elif retrieval_type == "fusion":
            base_retriever = FusionRetriever(self.vectorstore, self.bm25_index, k=k)
        else:
            raise ValueError(f"未知的检索类型: {retrieval_type}")

        # 应用重排序
        if retrieval_config.get("use_reranker", False):
            try:
                retriever = get_reranker(
                    base_retriever,
                    retrieval_config.get("rerank_model", "rerank-english-v3.0"),
                    retrieval_config.get("rerank_top_n", 5),
                )
            except Exception as e:
                logger.warning(f"重排序器初始化失败，使用基础检索器: {str(e)}")
                retriever = base_retriever
        else:
            retriever = base_retriever

        return retriever

    def _get_enhanced_retriever(self, base_retriever):
        """获取增强检索器"""
        query_config = self.config["query_enhancement"]
        enhancement_type = query_config["type"]

        if enhancement_type == "none" or self.llm is None:
            return base_retriever
        elif enhancement_type == "rewrite":
            return get_rewritten_query_retriever(base_retriever, self.llm)
        elif enhancement_type == "multi_query":
            return get_multi_query_retriever(base_retriever, self.llm)
        elif enhancement_type == "rag_fusion":
            return get_rag_fusion_retriever(base_retriever, self.llm)
        elif enhancement_type == "step_back":
            return get_step_back_retriever(base_retriever, self.llm)
        else:
            return base_retriever

    def query(
        self, question: str, chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """处理用户查询"""
        try:
            if not question.strip():
                return "请输入您的问题", chat_history

            if self.llm is None:
                return "❌ 请先初始化LLM", chat_history

            if self.vectorstore is None:
                return "❌ 请先索引文档", chat_history

            # 获取检索器
            base_retriever = self._get_retriever()
            enhanced_retriever = self._get_enhanced_retriever(base_retriever)

            # 语义路由（如果启用）
            if self.config["routing"]["enabled"]:
                try:
                    prompt_template = semantic_routing(question, self.config["routing"])
                except Exception as e:
                    logger.warning(f"语义路由失败，使用默认模板: {str(e)}")
                    prompt_template = ChatPromptTemplate.from_template(
                        "基于以下上下文回答问题:\n\n{context}\n\n问题: {question}\n\n回答:"
                    )
            else:
                prompt_template = ChatPromptTemplate.from_template(
                    "基于以下上下文回答问题:\n\n{context}\n\n问题: {question}\n\n回答:"
                )

            # 构建RAG链
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {
                    "context": enhanced_retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | self.llm
                | StrOutputParser()
            )

            # 执行查询
            response = rag_chain.invoke(question)

            # 更新聊天历史
            chat_history.append((question, response))

            logger.info(f"查询处理完成: {question[:50]}...")
            return response, chat_history

        except Exception as e:
            error_msg = f"❌ 查询处理失败: {str(e)}"
            logger.error(error_msg)
            chat_history.append((question, error_msg))
            return error_msg, chat_history

    def get_current_config(self) -> str:
        """获取当前配置"""
        return json.dumps(self.config, indent=2, ensure_ascii=False)

    def get_config_for_display(self) -> str:
        """获取用于UI显示的配置（隐藏敏感信息）"""
        display_config = json.loads(json.dumps(self.config))

        # 隐藏敏感字段
        if "llm" in display_config:
            if "api_key" in display_config["llm"]:
                display_config["llm"]["api_key"] = "***"
        if "embedding" in display_config:
            if "api_key" in display_config["embedding"]:
                display_config["embedding"]["api_key"] = "***"

        return json.dumps(display_config, indent=2, ensure_ascii=False)

    def clear_index(self) -> str:
        """清空索引"""
        self.vectorstore = None
        self.bm25_index = None
        self.indexed_files.clear()
        logger.info("索引已清空")
        return "✅ 索引已清空"


def create_interface():
    """创建Gradio界面"""
    app = RAGApplication()

    with gr.Blocks(title="RAG应用系统", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 RAG应用系统")
        gr.Markdown("一个功能完整的检索增强生成系统，支持多种RAG技术的配置和使用")

        with gr.Tabs():
            # 配置标签页
            with gr.Tab("⚙️ 系统配置"):
                gr.Markdown("### LLM配置")
                with gr.Row():
                    with gr.Column():
                        config_input = gr.Code(
                            value=app.get_config_for_display(),
                            language="json",
                            label="配置JSON",
                            lines=20,
                        )
                        update_config_btn = gr.Button("更新配置", variant="primary")
                        config_status = gr.Textbox(label="配置状态", interactive=False)

                    with gr.Column():
                        init_llm_btn = gr.Button("初始化LLM", variant="primary")
                        llm_status = gr.Textbox(label="LLM状态", interactive=False)

                        gr.Markdown("### 当前配置说明")
                        gr.Markdown("""
                        **支持的配置选项：**
                        - **LLM**: OpenAI兼容模式（通过API Base URL配置）
                        - **分块器**: recursive, token, SemanticSpacyChunker, SemanticNLTKChunker
                        - **检索类型**: similarity, bm25, fusion
                        - **查询增强**: none, rewrite, multi_query, rag_fusion, step_back
                        - **重排序**: 支持Cohere Rerank
                        """)

            # 文档管理标签页
            with gr.Tab("📁 文档管理"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="选择文档文件",
                            file_count="multiple",
                            file_types=[".pdf"],
                        )
                        index_btn = gr.Button("索引文档", variant="primary")
                        clear_btn = gr.Button("清空索引", variant="secondary")

                    with gr.Column():
                        index_status = gr.Textbox(label="索引状态", interactive=False)
                        indexed_files = gr.Textbox(
                            label="已索引文件", value="暂无", interactive=False, lines=5
                        )

            # 对话标签页
            with gr.Tab("💬 智能对话"):
                chatbot = gr.Chatbot(
                    label="对话历史", height=400, bubble_full_width=False
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：请介绍一下文档中的主要内容...",
                        scale=4,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                clear_chat_btn = gr.Button("清空对话", variant="secondary")

        # 事件绑定
        update_config_btn.click(
            app.update_config, inputs=[config_input], outputs=[config_status]
        )

        init_llm_btn.click(app.init_llm, outputs=[llm_status])

        index_btn.click(
            app.index_documents, inputs=[file_input], outputs=[index_status]
        ).then(
            lambda: "\n".join(app.indexed_files) if app.indexed_files else "暂无",
            outputs=[indexed_files],
        )

        clear_btn.click(app.clear_index, outputs=[index_status]).then(
            lambda: "暂无", outputs=[indexed_files]
        )

        submit_btn.click(
            app.query,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot],
        ).then(lambda: "", outputs=[question_input])

        question_input.submit(
            app.query,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot],
        ).then(lambda: "", outputs=[question_input])

        clear_chat_btn.click(lambda: [], outputs=[chatbot])

    return interface


def main():
    """主函数"""
    logger.info("启动RAG应用系统")

    # 创建必要的目录
    os.makedirs("./vector_store", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # 创建并启动界面
    interface = create_interface()
    interface.launch(
        share=False, inbrowser=True, server_name="0.0.0.0", server_port=7860
    )


if __name__ == "__main__":
    main()
