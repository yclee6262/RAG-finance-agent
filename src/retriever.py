import os
import shutil
import pickle  # <--- 新增：用於序列化物件
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore # <--- 新增 EncoderBackedStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import torch

load_dotenv()

# 設定儲存路徑
INDEX_PATH = "data/faiss_index"
DOCSTORE_PATH = "data/doc_store"

def get_embedding_model():
    """使用地端 BGE-M3 模型 (GPU 加速)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在載入地端 Embedding 模型 (BGE-M3) 至 {device}...")
    
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_and_save_retriever(documents):
    embedding_model = get_embedding_model()

    # 1. 定義切分器
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # 2. 初始化向量儲存 (FAISS)
    vectorstore = FAISS.from_documents(
        [Document(page_content="init", metadata={})], 
        embedding_model
    )
    
    # 3. 初始化文件儲存 (LocalFileStore + Encoder) <--- 關鍵修改
    if os.path.exists(DOCSTORE_PATH):
        shutil.rmtree(DOCSTORE_PATH)
    os.makedirs(DOCSTORE_PATH, exist_ok=True)
    
    # 建立基礎檔案儲存層
    fs = LocalFileStore(DOCSTORE_PATH)
    
    # 包裝一層 Encoder，負責把 Document 物件轉成 bytes (使用 pickle)
    docstore = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x, # key 不需要編碼
        value_serializer=pickle.dumps,    # 寫入時：物件 -> bytes
        value_deserializer=pickle.loads     # 讀取時：bytes -> 物件
    )

    # 4. 建立 Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 5. 加入文件
    print(f"正在建立索引並寫入硬碟...")
    retriever.add_documents(documents)
    
    # 6. 儲存 FAISS 索引
    vectorstore.save_local(INDEX_PATH)
    
    print(f"✅ 索引建立完成！")
    print(f" - 向量庫位置: {INDEX_PATH}")
    print(f" - 文件庫位置: {DOCSTORE_PATH}")

if __name__ == "__main__":
    # 讀取 Markdown
    md_path = "data/processed/bot_report_113_enriched.md"
    if not os.path.exists(md_path):
        md_path = "data/processed/bot_report_113.md" # fallback

    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content, metadata={"source": "113年報"})]
        build_and_save_retriever(docs)
    else:
        print("❌ 找不到 markdown 檔案，請先執行 parser.py")