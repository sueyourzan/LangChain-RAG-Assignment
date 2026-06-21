# compare_reranker.py —— 重排序效果对比
# 对比加入 BGE-Reranker 前后，检索结果 top-3 的相关性变化

# 必须在所有 import 之前设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import shutil
from eval_framework import (
    load_all_docs, split_docs, create_vector_db,
    load_llm, build_qa_chain, run_evaluation, print_comparison,
    TEST_QUESTIONS, start_log, end_log
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DB_DIR = "./chroma_db_reranker"

# BGE 重排序模型（CrossEncoder）
RERANKER_MODEL = "BAAI/bge-reranker-base"


def build_qa_chain_with_reranker(vector_db, llm, reranker, k_retrieve=10, k_final=3):
    """构建带重排序的问答链：先召回k_retrieve个，用reranker重排后取top k_final"""
    from langchain_classic.chains import RetrievalQA
    from langchain_core.retrievers import BaseRetriever

    retriever = vector_db.as_retriever(search_kwargs={"k": k_retrieve})

    class RerankedRetriever(BaseRetriever):
        """包装原始retriever，加入重排序逻辑"""
        base_retriever: object
        reranker: object
        retrieve_k: int
        final_k: int

        def _get_relevant_documents(self, query, run_manager=None):
            docs = self.base_retriever.invoke(query)
            if len(docs) <= self.final_k:
                return docs
            # 用 reranker 对每对 (query, doc) 打分
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            # 按分数降序排列
            scored_docs = list(zip(scores, docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:self.final_k]]

    reranked_retriever = RerankedRetriever(
        base_retriever=retriever, reranker=reranker,
        retrieve_k=k_retrieve, final_k=k_final
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=reranked_retriever,
        return_source_documents=True
    )
    return qa_chain


def main():
    writer = start_log("结果_重排序效果对比.txt")

    print("=" * 60)
    print("  实验3: 重排序效果对比 (BGE-Reranker)")
    print("=" * 60)

    documents = load_all_docs()
    if not documents:
        print("未读取到文档，退出")
        end_log(writer)
        return

    split_docs_result = split_docs(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # 清理旧库
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    vector_db = create_vector_db(split_docs_result, EMBED_MODEL, DB_DIR)

    print("\n正在加载LLM...")
    llm = load_llm()

    # 加载重排序模型
    print(f"\n正在加载重排序模型: {RERANKER_MODEL} ...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANKER_MODEL)
    print("重排序模型加载完成")

    # ---- 基础检索（无重排序）----
    print("\n>>> 基础检索（无重排序）")
    base_chain = build_qa_chain(vector_db, llm, k=3)
    base_results = run_evaluation(base_chain)

    # ---- 带重排序的检索 ----
    print("\n>>> 带重排序的检索（召回10条 → 重排 → Top3）")
    reranked_chain = build_qa_chain_with_reranker(vector_db, llm, reranker)
    reranked_results = run_evaluation(reranked_chain)

    # 对比输出
    print_comparison(base_results, reranked_results,
                     label_a="无重排序 (Top3)", label_b="有重排序 (召回10→重排→Top3)")

    # 分析重排序带来的变化
    print(f"\n{'='*60}")
    print("  重排序变化分析（检索文档是否更相关）")
    print(f"{'='*60}")
    for i in range(len(base_results)):
        base_docs = {d.page_content[:80] for d in base_results[i]["source_documents"]}
        rerank_docs = {d.page_content[:80] for d in reranked_results[i]["source_documents"]}
        new_docs = rerank_docs - base_docs
        lost_docs = base_docs - rerank_docs
        print(f"  问题{i+1}: 新增块={len(new_docs)}, 丢弃块={len(lost_docs)}")

    end_log(writer)


if __name__ == "__main__":
    main()
