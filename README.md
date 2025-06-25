# 🤖 RAG应用系统

一个功能完整的检索增强生成（Retrieval-Augmented Generation）系统，支持多种RAG技术的配置和使用，提供直观的Web界面。

## ✨ 项目特点

### 🔧 高度可配置
- **多种分块策略**：支持递归分块、语义分块、Token分块等
- **多样化检索方式**：向量检索、BM25、混合检索
- **查询增强技术**：查询重写、多查询、RAG Fusion、Step Back等
- **智能重排序**：集成Cohere Rerank模型
- **语义路由**：基于查询内容自动选择提示模板

### 🧠 多模型支持
- **LLM提供商**：OpenAI、Groq
- **嵌入模型**：HuggingFace Transformers系列
- **重排序模型**：Cohere系列

### 🎨 用户友好
- **可视化界面**：基于Gradio的Web UI
- **实时配置**：支持JSON配置动态更新
- **对话式交互**：类ChatGPT的聊天界面
- **文档管理**：拖拽上传，一键索引

## 📁 项目结构

```
rag/
├── main.py                 # 主应用入口，Gradio界面
├── pyproject.toml          # 项目依赖配置
├── README.md               # 项目说明文档
├── uv.lock                 # 依赖锁定文件
│
├── indexing/               # 文档索引模块
│   ├── __init__.py
│   ├── bm25_index.py       # BM25索引构建
│   ├── chunker.py          # 文本分块器（递归、语义等）
│   ├── data_processor.py   # 数据处理器（PDF等）
│   ├── embeddings.py       # 嵌入模型封装
│   ├── indexer.py          # 索引器主类
│   └── verctorstore.py     # 向量存储（FAISS）
│
├── llm/                    # 大语言模型模块
│   └── llm.py              # LLM工厂函数
│
├── mappers/                # 映射配置
│   └── mappers.py          # 组件映射关系
│
├── prompts/                # 提示模板（待扩展）
│
├── query/                  # 查询增强模块
│   └── query_translation.py # 查询重写、多查询等
│
├── reranking/              # 重排序模块
│   └── reranker.py         # Cohere重排序
│
├── retrieval/              # 检索模块
│   └── retriever.py        # 多种检索器实现
│
├── routing/                # 路由模块
│   └── routing.py          # 语义路由
│
└── utils/                  # 工具模块
    ├── logging.py          # 日志配置
    └── parser.py           # 输出解析器
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- uv包管理器（推荐）或pip

### 安装依赖

使用uv（推荐）：
```bash
# 安装uv
pip install uv

# 安装项目依赖
uv sync
```

或使用pip：
```bash
pip install -r requirements.txt
```

### 配置环境变量

创建`.env`文件并配置必要的API密钥：

```bash
# OpenAI配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# Groq配置（可选）
GROQ_API_KEY=your_groq_api_key

# Cohere配置（用于重排序，可选）
COHERE_API_KEY=your_cohere_api_key
```

### 运行应用

```bash
# 激活环境（如果使用uv）
uv run python main.py

# 或直接运行
python main.py
```

应用将在 `http://localhost:7860` 启动。

## 📖 使用指南

### 1. 配置系统

在"系统配置"标签页中：

1. **更新LLM配置**：修改JSON配置中的API密钥、模型等参数
2. **初始化LLM**：点击"初始化LLM"按钮

示例配置：
```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "your_api_key",
    "api_base": "https://api.openai.com/v1",
    "model_config": {
      "temperature": 0.7,
      "max_tokens": 1000
    }
  },
  "chunker": {
    "type": "recursive",
    "params": {
      "chunk_size": 512,
      "chunk_overlap": 64
    }
  },
  "retrieval": {
    "type": "fusion",
    "k": 5,
    "use_reranker": true
  },
  "query_enhancement": {
    "type": "rag_fusion"
  }
}
```

### 2. 上传和索引文档

在"文档管理"标签页中：

1. **上传文档**：支持PDF格式，可多选
2. **索引文档**：点击"索引文档"开始处理
3. **查看状态**：实时显示索引进度和已索引文件

### 3. 智能对话

在"智能对话"标签页中：

1. 输入问题并发送
2. 系统将自动检索相关文档并生成回答
3. 支持多轮对话，保持上下文

## 🔧 核心技术

### 分块策略

- **递归分块**：基于分隔符的层次化分块
- **Token分块**：基于tiktoken的精确分块
- **语义分块**：基于spaCy/NLTK的智能分块

### 检索技术

- **向量检索**：基于FAISS的相似度检索
- **BM25检索**：基于词频的稀疏检索
- **混合检索**：结合向量和BM25的融合检索

### 查询增强

- **查询重写**：使用LLM重写用户查询
- **多查询生成**：生成多个相关查询并合并结果
- **RAG Fusion**：使用RRF算法融合多查询结果
- **Step Back**：生成更抽象的查询获取背景知识

### 重排序

- **Cohere Rerank**：使用Cohere专用重排序模型

## 📊 支持的文件格式

目前支持：
- PDF文档

计划支持：
- Word文档（.docx）
- 文本文件（.txt）
- Markdown文件（.md）
- 网页（HTML）

## 🔍 配置详解

### LLM配置
```json
{
  "provider": "openai|groq",
  "model": "模型名称",
  "api_key": "API密钥",
  "api_base": "API端点",
  "model_config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

### 分块器配置
```json
{
  "type": "recursive|token|SemanticSpacyChunker|SemanticNLTKChunker",
  "params": {
    "chunk_size": 512,
    "chunk_overlap": 64
  }
}
```

### 检索配置
```json
{
  "type": "similarity|bm25|fusion",
  "k": 5,
  "use_reranker": true,
  "rerank_model": "rerank-english-v3.0",
  "rerank_top_n": 5
}
```

### 查询增强配置
```json
{
  "type": "none|rewrite|multi_query|rag_fusion|step_back"
}
```

## 🐛 故障排除

### 常见问题

1. **API密钥错误**
   - 检查`.env`文件中的API密钥配置
   - 确保API密钥有效且有足够额度

2. **文档索引失败**
   - 检查文档格式是否支持
   - 确保文档没有密码保护或损坏

3. **内存不足**
   - 减小chunk_size参数
   - 减少同时处理的文档数量

4. **依赖安装失败**
   - 使用国内镜像源：`uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple`

## 📝 日志

应用日志存储在`./logs/`目录下，按日期命名：
- `rag_YYYYMMDD.log`：应用运行日志
- 包含详细的错误信息和调试信息

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 🌟 致谢

感谢以下开源项目：
- [LangChain](https://github.com/langchain-ai/langchain)
- [Gradio](https://github.com/gradio-app/gradio)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face](https://huggingface.co/)

---

## 🚧 未来计划

### 短期目标（1-2个月）
<!-- 在这里填写您的短期开发计划 -->

### 中期目标（3-6个月）
<!-- 在这里填写您的中期开发计划 -->

### 长期愿景（6个月以上）
<!-- 在这里填写您的长期发展愿景 -->

### 待实现功能
<!-- 在这里列出具体的待实现功能清单 -->

### 性能优化计划
<!-- 在这里描述性能优化的具体计划 -->

### 用户体验改进
<!-- 在这里描述UI/UX改进计划 -->

---

如有任何问题或建议，请随时联系我们！