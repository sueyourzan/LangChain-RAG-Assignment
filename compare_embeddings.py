# compare_embeddings.py —— 嵌入模型对比
# 对比 all-MiniLM-L6-v2 vs shibing624/text2vec-base-chinese

# 必须在所有 import 之前设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import shutil
from eval_framework import (
    load_all_docs, split_docs, create_vector_db,
    load_llm, build_qa_chain, run_evaluation, print_comparison,
    start_log, end_log
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBED_CONFIGS = [
    {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "all-MiniLM-L6-v2 (英文优化)",
        "db_dir": "./chroma_db_embed_minilm"
    },
    {
        "model_name": "shibing624/text2vec-base-chinese",
        "label": "text2vec-base-chinese (中文优化)",
        "db_dir": "./chroma_db_embed_text2vec"
    },
]


def main():
    writer = start_log("结果_嵌入模型对比.txt")

    print("=" * 60)
    print("  实验2: 嵌入模型对比 (MiniLM vs text2vec-chinese)")
    print("=" * 60)

    documents = load_all_docs()
    if not documents:
        print("未读取到文档，退出")
        end_log(writer)
        return

    split_docs_result = split_docs(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    print("\n正在加载LLM...")
    llm = load_llm()

    all_results = []
    for cfg in EMBED_CONFIGS:
        print(f"\n{'='*40}")
        print(f"  嵌入模型: {cfg['label']}")
        print(f"{'='*40}")

        if os.path.exists(cfg["db_dir"]):
            shutil.rmtree(cfg["db_dir"])

        vector_db = create_vector_db(split_docs_result, cfg["model_name"], cfg["db_dir"])
        qa_chain = build_qa_chain(vector_db, llm)
        results = run_evaluation(qa_chain)
        all_results.append((cfg["label"], results))

    # 对比打印
    print_comparison(all_results[0][1], all_results[1][1],
                     label_a=all_results[0][0], label_b=all_results[1][0])

    # 统计检索文档重叠度
    print(f"\n{'='*60}")
    print("  检索文档重叠度分析")
    print(f"{'='*60}")
    for i in range(len(all_results[0][1])):
        docs_a = {d.page_content[:100] for d in all_results[0][1][i]["source_documents"]}
        docs_b = {d.page_content[:100] for d in all_results[1][1][i]["source_documents"]}
        overlap = docs_a & docs_b
        print(f"  问题{i+1}: 重叠块数 = {len(overlap)} / "
              f"模型A={len(docs_a)}, 模型B={len(docs_b)}")

    end_log(writer)


if __name__ == "__main__":
    main()
