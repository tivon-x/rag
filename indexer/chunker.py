#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from langchain.text_splitter import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
from langchain.docstore.document import Document
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import jieba


class Chunker(ABC):
    @abstractmethod
    def chunk(self, docs: list[Document]) -> list[Document]:
        """
        分块处理文档
        Args:
            docs (List[Document]): 输入文档列表
        Returns:
            list[Document]: 分块后的文档列表
        """
        pass

class RecursiveChunker(Chunker):
    """基于递归字符分割的文本分块器"""
    def __init__(self, chunk_size=512, chunk_overlap=64):
        """
        初始化分块器
        Args:
            chunk_size (int): 每个分块的最大字符数
            chunk_overlap (int): 分块之间的重叠字符数
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n# ", "\n\n", "\n", "。", "!", "?", " ", ""]
        ) # 注意，它会保留metadata信息

    def chunk(self, docs: list[Document]) -> list[Document]:
        return self.splitter.split_documents(docs)
    

class TokenChunker(Chunker):
    """基于字符编码的递归文本分块器"""
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def chunk(self, docs: list[Document]) -> list[Document]:
        return self.splitter.split_documents(docs)


class SemanticSpacyChunker(Chunker):
    """基于spaCy语义分析的智能文本分割器"""
    def __init__(
        self,
        model_name: str = "zh_core_web_sm",  # 支持中英文模型切换： en_core_web_sm
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_sentence: bool = True  # 是否基于句子拆分
    ):
        self.nlp = spacy.load(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_sentence = use_sentence

    def split_text(self, doc: Document) -> list[Document]:
        """核心分割逻辑"""
        doc_text = self.nlp(doc.page_content)

        if self.use_sentence:
            sentences = [sent.text for sent in doc_text.sents]
        else:
            sentences = [token.text for token in doc_text if not token.is_punct]
        # 动态合并句子/词块
        current_chunk = []
        current_length = 0
        chunks = []

        for sent in sentences:
            sent_length = len(sent.split(' '))  # 按空格分词计算长度
            
            # 判断是否超过阈值
            if current_length + sent_length > self.chunk_size:
                if current_chunk:
                    # 中文用空字符串连接
                    chunks.append("".join(current_chunk))
                    
                    # 精确计算重叠字符数
                    overlap_buffer = []
                    overlap_length = 0
                    # 逆向遍历寻找重叠边界
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) > self.chunk_overlap:
                            break
                        overlap_buffer.append(s)
                        overlap_length += len(s)
                    # 恢复原始顺序
                    current_chunk = list(reversed(overlap_buffer))
                    current_length = overlap_length
                    
            current_chunk.append(sent)
            current_length += sent_length

        # 处理剩余内容
        if current_chunk:
            chunks.append("".join(current_chunk))
        return [Document(page_content=chunk, metadata=doc.metadata.copy()) for chunk in chunks]

    def chunk(self, docs: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        for doc in docs:
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks


class SemanticNLTKChunker(Chunker):
    """基于NLTK的智能语义分块器，支持中英文混合文本"""
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        language: str = "chinese",
        use_jieba: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.use_jieba = use_jieba

        # 初始化中文分词器
        if self.language == "chinese" and self.use_jieba:
            jieba.initialize()
    def _chinese_sentence_split(self, text: str) -> list[str]:
        """基于结巴分词的智能分句"""
        if not self.use_jieba:
            return [text]
            
        delimiters = {'。', '！', '？', '；', '…'}
        sentences = []
        buffer = []
        
        for word in jieba.cut(text):
            buffer.append(word)
            if word in delimiters:
                sentences.append(''.join(buffer))
                buffer = []
        
        if buffer:  # 处理末尾无标点的句子
            sentences.append(''.join(buffer))
        return sentences

    def split_text(self, doc: Document) -> list[Document]:
        """多语言分句逻辑"""
        sentences = []
        if self.language == "chinese":
            sentences =  self._chinese_sentence_split(doc.page_content)
        else:
            nltk.download('punkt_tab')
            sentences =  sent_tokenize(doc.page_content, language=self.language)

        """动态合并句子并保留字符重叠"""
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_buffer = []

        for sent in sentences:
            sent_len = len(sent.split(' '))  # 按空格分词计算长度
            
            # 触发分块条件
            if current_length + sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    
                    # 计算重叠部分
                    overlap_buffer = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) > self.chunk_overlap:
                            break
                        overlap_buffer.append(s)
                        overlap_length += len(s)
                        
                    current_chunk = list(reversed(overlap_buffer))
                    current_length = overlap_length
            
            current_chunk.append(sent)
            current_length += sent_len

        # 处理剩余内容
        if current_chunk:
            chunks.append("".join(current_chunk))
        return [Document(page_content=chunk, metadata=doc.metadata.copy()) for chunk in chunks]

    def chunk(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in docs:
            chunks.extend(self.split_text(doc))
        return chunks
    