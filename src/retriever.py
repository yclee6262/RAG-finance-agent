import os
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# 設定向量儲存路徑 (因為 FAISS 是檔案，要指定存哪)
INDEX_PATH = "data/faiss_index"

def get_embedding_model():
    """取得 Azure OpenAI Embedding 模型"""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

def build_and_save_retriever(documents):
    """
    建立 Parent-Child Retriever 並使用 FAISS 作為向量存儲
    """
    print(f"正在建立索引，共 {len(documents)} 份文件...")
    
    embedding_model = get_embedding_model()

    # 1. 定義切分器 (關鍵：金融財報需要精細切分)
    # Child: 針對具體數據 (如: "營收成長 5%")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    # Parent: 針對上下文 (如: 整頁資產負債表)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # 2. 初始化向量儲存 (Child Vectors) - 使用 FAISS
    # 注意：ParentDocumentRetriever 需要一個空的 vectorstore 開始
    vectorstore = FAISS.from_documents(
        [Document(page_content="init", metadata={})], # 初始化用假資料
        embedding_model
    )
    
    # 3. 初始化文件儲存 (Parent Documents) - 這裡先用記憶體，正式可改用 LocalFileStore
    docstore = InMemoryStore()

    # 4. 建立 Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 5. 加入真實文件 (這步最花時間，會呼叫 Embedding API)
    retriever.add_documents(documents)
    
    # 6. 儲存 FAISS 索引 (重要！FAISS 不會自動存)
    print(f"正在儲存索引至 {INDEX_PATH}...")
    vectorstore.save_local(INDEX_PATH)
    
    print("索引建立完成！")
    return retriever

def load_retriever():
    """
    讀取已儲存的 FAISS 索引
    """
    embedding_model = get_embedding_model()
    
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("找不到索引檔案，請先執行建立索引的步驟。")

    # 載入 FAISS
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embedding_model,
        allow_dangerous_deserialization=True # 本地實驗通常開啟此選項
    )
    
    # 重新組裝 Retriever (注意：InMemoryStore 重新執行時會清空，
    # 如果要持久化 Parent Docs，需要改用 LocalFileStore，
    # 但論文實驗通常每次重跑沒關係)
    docstore = InMemoryStore() 
    
    # 這裡有個小技巧：如果是讀取舊索引，通常只需拿來做 QA，
    # 若要繼續用 ParentDocumentRetriever 的完整功能，需要把 docstore 也持久化。
    # 為了簡化 Phase 1，我們先回傳 vectorstore 即可進行相似度搜尋測試。
    return vectorstore

if __name__ == "__main__":    
    # 1. 讀取 Markdown (假設你上一步已經轉好了)
    md_path = "data/processed/bot_report_113.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs = [Document(page_content=content, metadata={"source": "113年報"})]
        
        # 2. 建立索引
        build_and_save_retriever(docs)
    else:
        print("請先執行 parser.py 產生 markdown 檔案")