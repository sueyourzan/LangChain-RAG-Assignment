# eval_framework.py —— 公共评估框架
# 提供文档加载、LLM加载、评估运行、结果打印等共享功能

# 必须在所有 import 之前设置，否则 sentence-transformers 会直连 huggingface.co
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==================== 配置 ====================
FILE_PATHS = [
    "大模型基础概念.txt",
    "检索增强生成RAG技术.txt",
    "大模型优化与落地应用.txt"
]

# 3个测试问题
TEST_QUESTIONS = [
    "什么是大语言模型？它的核心架构是什么？",
    "RAG系统的核心工作模块有哪些？",
    "传统大模型存在哪些局限性？",
]

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==================== 公共函数 ====================

def load_all_docs(file_list=None):
    """加载多个TXT文档"""
    if file_list is None:
        file_list = FILE_PATHS
    all_documents = []
    for fp in file_list:
        if not os.path.exists(fp):
            print(f"警告：文件 {fp} 不存在！")
            continue
        loader = TextLoader(fp, encoding="utf-8")
        all_documents.extend(loader.load())
        print(f"成功加载文件：{fp}")
    return all_documents


def split_docs(docs, chunk_size=800, chunk_overlap=150):
    """文本切分"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    split_docs = splitter.split_documents(docs)
    print(f"切分完成，共 {len(split_docs)} 个文本块 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return split_docs


def create_vector_db(split_docs, embed_model_name, db_dir):
    """构建/加载向量数据库"""
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    if os.path.exists(db_dir):
        vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        print(f"加载已有向量库: {db_dir}")
    else:
        vector_db = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=db_dir
        )
        vector_db.persist()
        print(f"向量库创建完成: {db_dir}")
    return vector_db


def load_llm(model_name=None):
    """加载本地大模型"""
    if model_name is None:
        model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=300, temperature=0.2, top_p=0.9, do_sample=True,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_qa_chain(vector_db, llm, k=3):
    """构建检索问答链"""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


def run_evaluation(qa_chain, questions=None):
    """逐题运行评估，返回结果列表"""
    if questions is None:
        questions = TEST_QUESTIONS
    results = []
    for i, q in enumerate(questions):
        print(f"  评测 [{i+1}/{len(questions)}]: {q[:50]}...")
        result = qa_chain.invoke(q)
        results.append({
            "question": q,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        })
    return results


def print_results(results, title="评估结果"):
    """格式化打印评估结果"""
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    for i, r in enumerate(results):
        print(f"\n{'═'*60}")
        print(f"  Human: {r['question']}")
        print(f"{'═'*60}")
        print(f"  上下文 ({len(r['source_documents'])}块):")
        print(f"{'─'*60}")
        for j, doc in enumerate(r['source_documents']):
            content = doc.page_content.replace('\n', ' ')[:200]
            print(f"    [{j+1}] {content}")
        print(f"{'─'*60}")
        print(f"  回答:")
        print(f"    {r['answer'][:400]}")
        print(f"{'═'*60}")


def print_comparison(results_a, results_b, label_a="策略A", label_b="策略B"):
    """并排打印两种策略的对比结果"""
    print(f"\n{'═'*60}")
    print(f"  对比: {label_a}  vs  {label_b}")
    print(f"{'═'*60}")
    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        print(f"\n{'═'*60}")
        print(f"  Human: {ra['question']}")
        print(f"{'═'*60}")
        print(f"  [{label_a}] 上下文 ({len(ra['source_documents'])}块):")
        print(f"{'─'*60}")
        for j, doc in enumerate(ra['source_documents']):
            print(f"    [{j+1}] {doc.page_content.replace(chr(10), ' ')[:150]}")
        print(f"  [{label_a}] 回答: {ra['answer'][:300]}")
        print(f"{'·'*60}")
        print(f"  [{label_b}] 上下文 ({len(rb['source_documents'])}块):")
        print(f"{'─'*60}")
        for j, doc in enumerate(rb['source_documents']):
            print(f"    [{j+1}] {doc.page_content.replace(chr(10), ' ')[:150]}")
        print(f"  [{label_b}] 回答: {rb['answer'][:300]}")
        print(f"{'═'*60}")


# ==================== 日志工具 ====================

class DualWriter:
    """同时输出到终端和文件"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.terminal


def start_log(file_path):
    """开始记录日志到文件（同时在终端显示）"""
    writer = DualWriter(file_path)
    sys.stdout = writer
    return writer


def end_log(writer):
    """结束日志记录"""
    writer.close()
    print(f"\n结果已保存到: {writer.file.name}")
