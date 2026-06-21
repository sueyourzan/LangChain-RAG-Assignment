# 必须在所有 import 之前设置，否则 sentence-transformers 会直连 huggingface.co
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 导入依赖库
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==================== 配置项 ====================
file_paths = [
    "大模型基础概念.txt",
    "检索增强生成RAG技术.txt",
    "大模型优化与落地应用.txt"
]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
VECTOR_DB_DIR = "./chroma_db"

# 轻量嵌入模型 + 轻量对话模型（国内镜像加速下载）
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# 加载多个TXT文档
def load_all_docs(file_list):
    all_documents = []
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在！")
            continue
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        all_documents.extend(docs)
        print(f"成功加载文件：{file_path}")
    return all_documents

# 文本切分
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"文本切分完成，共 {len(split_docs)} 个文本块")
    return split_docs

# 构建向量数据库
def create_vector_db(split_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    if os.path.exists(VECTOR_DB_DIR):
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("加载已有向量库")
    else:
        vector_db = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vector_db.persist()
        print("向量库创建完成")
    return vector_db

# 加载本地大模型（适配普通Windows电脑，修复警告）
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# 构建问答链
def build_rag_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "你是一个AI助手。请根据以下上下文信息回答用户的问题。"
        "如果上下文中没有相关信息，请如实说明你不知道。"
        "请用中文回答。\n\n上下文：{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("system", system_prompt)
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# 主程序
if __name__ == "__main__":
    print("===== 启动 RAG 问答系统 =====")
    documents = load_all_docs(file_paths)
    if not documents:
        print("未读取到文档，退出")
        exit()
    split_docs = split_documents(documents)
    vector_database = create_vector_db(split_docs)

    print("正在加载模型，国内镜像加速中，请稍等...")
    llm_model = load_llm()
    rag_qa = build_rag_chain(vector_database, llm_model)
    print("===== 系统就绪，输入问题提问，输入 exit 退出 =====")

    while True:
        question = input("\n请输入问题：")
        if question.lower() == "exit":
            print("程序结束")
            break
        result = rag_qa.invoke({"input": question})
        print(f"\n{'═'*50}")
        print(f"  Human: {question}")
        print(f"{'═'*50}")
        print(f"  上下文 ({len(result['context'])}块):")
        print(f"{'─'*50}")
        for idx, doc in enumerate(result["context"]):
            print(f"  [{idx+1}] {doc.page_content[:200].replace(chr(10), ' ')}")
        print(f"{'─'*50}")
        print(f"  回答:")
        print(f"  {result['answer']}")
        print(f"{'═'*50}")