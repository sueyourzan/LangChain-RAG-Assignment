# compare_retrieval.py —— 检索优化策略对比
# 实现多查询融合（Multi-Query Fusion）检索，对比与基础单查询检索的差异

# 必须在所有 import 之前设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import shutil
from eval_framework import (
    load_all_docs, split_docs, create_vector_db,
    load_llm, build_qa_chain, run_evaluation, print_comparison,
    TEST_QUESTIONS, MODEL_NAME, start_log, end_log
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DB_DIR = "./chroma_db_retrieval"


def generate_query_variations(llm, question, n=3):
    """用LLM生成同一问题的多个变体查询"""
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.chains import LLMChain

    prompt = PromptTemplate(
        input_variables=["n", "question"],
        template="""Generate {n} different rephrased versions of the following question in Chinese.
Each version should express the same meaning using different wording.
Output only the numbered questions, one per line.

Original question: {question}

Rephrased versions:
1."""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"n": n, "question": question})

    # 解析返回的变体
    variations = [question]  # 保留原始问题
    text = response["text"] if isinstance(response, dict) else str(response)
    for line in text.strip().split("\n"):
        line = line.strip()
        # 去掉编号前缀
        if line and (line[0].isdigit()):
            # 尝试提取编号后的文本
            for sep in [". ", ".", "：", ": ", ":", "、"]:
                if sep in line:
                    line = line.split(sep, 1)[-1].strip()
                    break
            if len(line) > 5:
                variations.append(line)
    return variations


def multi_query_retrieve(vector_db, questions, k_per_query=3):
    """多查询融合检索：每个变体各自检索，合并去重后返回"""
    retriever = vector_db.as_retriever(search_kwargs={"k": k_per_query})

    all_docs = []
    seen = set()
    for q in questions:
        docs = retriever.invoke(q)
        for doc in docs:
            key = doc.page_content[:120]  # 用前120字符做去重标记
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
    return all_docs


def build_multi_query_chain(vector_db, llm):
    """构建多查询融合问答链"""
    from langchain_classic.chains import RetrievalQA
    from langchain_core.retrievers import BaseRetriever

    class MultiQueryRetriever(BaseRetriever):
        vec_db: object
        query_llm: object
        k_per_query: int = 3

        def _get_relevant_documents(self, query, run_manager=None):
            # 生成查询变体
            variations = generate_query_variations(self.query_llm, query, n=3)
            print(f"    查询变体数: {len(variations)}")
            for v in variations:
                print(f"      - {v[:60]}...")
            # 多查询融合检索
            docs = multi_query_retrieve(self.vec_db, variations, self.k_per_query)
            print(f"    融合后文档数: {len(docs)}")
            return docs[:5]  # 限制最终文档数

    retriever = MultiQueryRetriever(vec_db=vector_db, query_llm=llm, k_per_query=3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


def main():
    writer = start_log("结果_检索优化对比.txt")

    print("=" * 60)
    print("  实验4: 检索优化策略对比 (多查询融合 vs 单查询)")
    print("=" * 60)

    documents = load_all_docs()
    if not documents:
        print("未读取到文档，退出")
        end_log(writer)
        return

    split_docs_result = split_docs(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    vector_db = create_vector_db(split_docs_result, EMBED_MODEL, DB_DIR)

    print("\n正在加载LLM（同时用于查询改写和生成）...")
    llm = load_llm()

    # ---- 基础单查询检索 ----
    print("\n>>> 基础单查询检索")
    base_chain = build_qa_chain(vector_db, llm, k=3)
    base_results = run_evaluation(base_chain)

    # ---- 多查询融合检索 ----
    print("\n>>> 多查询融合检索")
    multi_chain = build_multi_query_chain(vector_db, llm)
    multi_results = run_evaluation(multi_chain)

    # 对比输出
    print_comparison(base_results, multi_results,
                     label_a="单查询检索 (Top3)", label_b="多查询融合检索 (Top5)")

    # 检索召回量对比
    print(f"\n{'='*60}")
    print("  检索召回量对比")
    print(f"{'='*60}")
    for i in range(len(base_results)):
        base_cnt = len(base_results[i]["source_documents"])
        multi_cnt = len(multi_results[i]["source_documents"])
        print(f"  问题{i+1}: 单查询={base_cnt}块, 多查询融合={multi_cnt}块")

    end_log(writer)


if __name__ == "__main__":
    main()
