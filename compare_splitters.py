# compare_splitters.py —— 文本切分策略对比
# 对比不同 chunk_size（400 vs 800 vs 1200）对检索和生成的影响

# 必须在所有 import 之前设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import shutil
from eval_framework import (
    load_all_docs, split_docs, create_vector_db,
    load_llm, build_qa_chain, run_evaluation, print_comparison,
    print_results, start_log, end_log
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 三种切分配置
SPLIT_CONFIGS = [
    {"chunk_size": 400, "chunk_overlap": 80,  "label": "chunk=400",   "db_dir": "./chroma_db_split_400"},
    {"chunk_size": 800, "chunk_overlap": 150, "label": "chunk=800",   "db_dir": "./chroma_db_split_800"},
    {"chunk_size": 1200,"chunk_overlap": 200, "label": "chunk=1200",  "db_dir": "./chroma_db_split_1200"},
]


def main():
    writer = start_log("结果_切分策略对比.txt")

    print("=" * 60)
    print("  实验1: 文本切分策略对比 (chunk_size: 400 vs 800 vs 1200)")
    print("=" * 60)

    documents = load_all_docs()
    if not documents:
        print("未读取到文档，退出")
        end_log(writer)
        return

    print("\n正在加载LLM...")
    llm = load_llm()

    all_results = []
    for cfg in SPLIT_CONFIGS:
        print(f"\n{'='*40}")
        print(f"  策略: {cfg['label']}")
        print(f"{'='*40}")

        # 清理旧库，确保每次重新构建
        if os.path.exists(cfg["db_dir"]):
            shutil.rmtree(cfg["db_dir"])

        split_docs_result = split_docs(documents, cfg["chunk_size"], cfg["chunk_overlap"])
        vector_db = create_vector_db(split_docs_result, EMBED_MODEL, cfg["db_dir"])
        qa_chain = build_qa_chain(vector_db, llm)
        results = run_evaluation(qa_chain)
        all_results.append((cfg["label"], results))

    # 打印所有策略的结果
    for label, results in all_results:
        print_results(results, f"切分策略: {label}")

    # 两两对比摘要
    print(f"\n{'='*60}")
    print("  切分策略对比摘要")
    print(f"{'='*60}")
    for i, q in enumerate(all_results[0][1]):
        print(f"\n【问题{i+1}】{q['question']}")
        for label, results in all_results:
            print(f"  [{label}] 检索块数={len(results[i]['source_documents'])}, "
                  f"回答={results[i]['answer'][:120]}...")

    end_log(writer)


if __name__ == "__main__":
    main()
