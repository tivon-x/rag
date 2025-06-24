#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List
import os

import numpy as np
from .chunker import Chunker
from .Embedder import Embedder
from Mappers.Mappers import LOADER_MAPPING, CHUNER_MAPPING, EMBEDDER_MAPPING


class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self.Chunker = None
        self.DocEmbedder = None
        self._init_components()

    def _init_components(self):
        """
        初始化索引器的组件，包括分块器和文档嵌入器。
        """

        # 初始化分块器
        chunker_cfg = self.config.get("chunker", {})
        self.Chunker = self._get_chunker(chunker_cfg)

        # 初始化文档嵌入器
        embedder_cfg = self.config.get("embedder", {})
        self.DocEmbedder = self._get_Embedder(embedder_cfg)

    def _get_data_processor(self, file_path: str) -> List[tuple[str, str]]:
        """
        根据文件路径获取数据处理器，返回处理后的数据和文件类型。
        Args:
          file_path (str): 文件路径

        Returns:
          List[tuple[str, str]]: 处理后的数据和文件类型的列表
        """

        # 递归处理文件夹
        if os.path.isdir(file_path):
            results = []
            for filename in os.listdir(file_path):
                # 跳过隐藏文件
                if filename.startswith('.'):
                    continue
                full_path = os.path.join(file_path, filename)
                try:
                    # 如果是文件，则表明接下来要处理
                    if os.path.isfile(full_path):
                        print("处理文件，路径为:", full_path)
                    results += self._get_data_processor(full_path)
                except ValueError as e:
                    print(f"跳过 {full_path}: {str(e)}")
                    continue
            return results
        # 如果是文件，则根据后缀名选择处理器
        else:
            ext = Path(file_path).suffix.lower() # 获取文件后缀名
            # 根据后缀名选择处理器
            loader_mapping = LOADER_MAPPING.get(ext)
            if loader_mapping is None:
                return []
            processor, loader_args = loader_mapping
            # 返回处理后的文件 + 后缀用来表示是图像还是文本
            return [(processor(**loader_args).process(file_path), file_path.split('.')[-1])]


    def _get_chunker(self, config: dict) -> Chunker:
        """
        获取分块器实例。
        Args:
            config (dict): 分块器配置
        Returns:
            Chunker: 分块器实例
        """
        # 获取分块器类型和参数
        chunker_type = config.get("type", "recursive") # 默认使用递归分块器
        params = config.get("params", {})
        chunker = CHUNER_MAPPING.get(chunker_type)
        if chunker is None:
            raise ValueError(f"Indexer_get_chunker -> 未知分块器类型: {chunker_type}")
        # 实例化
        return chunker(**params)

    def _get_Embedder(self, config: dict) -> Embedder:
        """
        获取文档嵌入器实例。
        Args:
            config (dict): 文档嵌入器配置
        Returns:
            Embedder: 文档嵌入器实例
        """
        docEmbedder_config = config.get("docEmbedder", {})
        docEmbedder_type = docEmbedder_config.get("type", "BAAIEmbedder")
        docParams = docEmbedder_config.get("params", {})
        docEmbedder = EMBEDDER_MAPPING.get(docEmbedder_type)
        # 实例化
        if docEmbedder is None:
            raise ValueError(f"Indexer_get_Embedder -> 未知嵌入器类型: {docEmbedder_type}")
        return docEmbedder(**docParams)

    def index(self, file_path: str) -> tuple[None | np.ndarray, str]:
        """
        索引文件
        Args:
            file_path (str): 文件路径

        Returns:
            tuple[None | np.ndarray, str]: 返回文档嵌入和分块数据
        """
        datas = self._get_data_processor(file_path)
        chunks = []
        for (data, type) in datas:
            if type in ['jpg', 'jpeg', 'png']:
                pass # 多模态信息除了图像就是文本, 先不处理图像
            else:
                chunks += self.Chunker.chunk(data)
        
        if self.DocEmbedder is not None:
            docEmb = self.DocEmbedder.embed(chunks)
        else:
            docEmb = None
        # 返回文档嵌入和分块数据
        return docEmb, chunks
        
        

    
    
