# LangChain RAG 实验项目

基于 LangChain 框架的检索增强生成（RAG）系统，包含基础 RAG 搭建和 4 项对比实验：文本切分策略、嵌入模型、重排序、检索优化。

---

## 项目结构

```
LangChain/
├── base_rag.py                  # 任务一：基础 RAG 问答系统
├── eval_framework.py            # 公共评估框架（文档加载、LLM、评估、日志）
├── compare_splitters.py         # 任务二-①：文本切分策略对比
├── compare_embeddings.py        # 任务二-②：嵌入模型对比
├── compare_reranker.py          # 任务二-③：重排序效果对比
├── compare_retrieval.py         # 任务二-④：检索优化策略对比
├── 大模型基础概念.txt            # 实验文档 1
├── 检索增强生成RAG技术.txt       # 实验文档 2
├── 大模型优化与落地应用.txt       # 实验文档 3
├── 结果_切分策略对比.txt          # 输出：切分实验
├── 结果_嵌入模型对比.txt          # 输出：嵌入实验
├── 结果_重排序效果对比.txt        # 输出：重排序实验
├── 结果_检索优化对比.txt          # 输出：检索实验
└── chroma_db_*/                 # 各实验的 Chroma 向量库
```

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n langchain_rag python=3.12 -y
conda activate langchain_rag
```

### 2. 安装依赖

```bash
pip install langchain langchain-community langchain-text-splitters \
    langchain-huggingface langchain-classic transformers \
    sentence-transformers chromadb \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 设置 HuggingFace 镜像

脚本内部已设置 `HF_ENDPOINT=https://hf-mirror.com`，首次运行会自动从镜像下载模型，无需科学上网。

---

## 模型一览

| 模型 | 用途 | 维度 | 大小 |
|------|------|------|------|
| `all-MiniLM-L6-v2` | 文本嵌入（默认） | 384 | ~90 MB |
| `shibing624/text2vec-base-chinese` | 文本嵌入（中文对比） | 768 | ~400 MB |
| `BAAI/bge-reranker-base` | 重排序 | — | ~1.1 GB |
| `Qwen/Qwen2-1.5B-Instruct` | 问答生成 | — | ~3 GB |

全部 CPU 推理，无需 GPU。

---

## 运行

### 任务一：基础 RAG 系统

```bash
python base_rag.py
```

- 加载 3 个 TXT 文档 → 切分 (chunk=800, overlap=150) → 向量库 (Chroma) → 交互式问答
- 输入 `exit` 退出

### 任务二：对比实验

| 实验 | 命令 | 对比内容 |
|------|------|----------|
| ① 切分策略 | `python compare_splitters.py` | chunk_size = 400 / 800 / 1200 |
| ② 嵌入模型 | `python compare_embeddings.py` | MiniLM (英文优化) vs text2vec (中文优化) |
| ③ 重排序 | `python compare_reranker.py` | 无重排序 Top-3 vs bge-reranker 重排 Top-3 |
| ④ 检索优化 | `python compare_retrieval.py` | 单查询检索 vs 多查询融合 (LLM 生成 3 变体) |

---

## 核心设计

### RAG 流水线

```
用户问题 → 文本嵌入 → 向量检索(Top-K) → [重排序] → 拼接上下文 → LLM 生成回答
```

### 公共评估框架 (`eval_framework.py`)

所有对比实验复用的基础设施：

- `load_all_docs()` — 文档加载
- `split_docs()` — 可配置 chunk_size / overlap 的切分
- `create_vector_db()` — Chroma 向量库构建 / 加载
- `load_llm()` — Qwen2-1.5B 本地推理
- `run_evaluation()` — 批量运行测试问题
- `print_comparison()` — 并排对比输出
- `DualWriter` — 终端 + 文件双路日志

---

## 测试问题

```python
TEST_QUESTIONS = [
    "什么是大语言模型？Transformer架构的核心机制是什么？",
    "RAG技术由哪几个核心环节组成？每个环节的作用是什么？",
    "大模型在实际落地中面临哪些挑战？有哪些优化手段？",
    "模型幻觉是什么？RAG如何帮助减少幻觉？",
    "模型轻量化有哪些常用技术？各自的特点是什么？",
]
```

---

## 输出示例

```
══════════════════════════════════════════════════
  Human: 什么是大语言模型？
══════════════════════════════════════════════════
  上下文 (3块):
──────────────────────────────────────────────────
  [1] 大语言模型（Large Language Model，LLM）是...
  [2] Transformer架构中的自注意力机制是...
  [3] 大模型的工作流程主要分为...
──────────────────────────────────────────────────
  回答:
  大语言模型是基于Transformer架构训练的大规模预训练模型...
══════════════════════════════════════════════════
```

---

## 常见问题

<details>
<summary><b>Q: 下载模型超时？</b></summary>

确认 `hf-mirror.com` 可访问：
```powershell
curl https://hf-mirror.com
```
</details>

<details>
<summary><b>Q: 显存 / 内存不足？</b></summary>

所有模型均支持 CPU 推理。若内存 < 8 GB，可换 `Qwen2-0.5B-Instruct`。
</details>

<details>
<summary><b>Q: VS Code 如何切换 Python 解释器？</b></summary>

`Ctrl+Shift+P` → `Python: Select Interpreter` → 选择 `D:\Conda_Envs\langchain_rag\python.exe`
</details>

---

## License

MIT
