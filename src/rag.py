import os
import torch
import pickle # <--- 新增
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore # <--- 新增
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 設定路徑
INDEX_PATH = "data/faiss_index"
DOCSTORE_PATH = "data/doc_store"

def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

def load_system():
    """載入 RAG 系統 (Retriever + LLM)"""
    print("正在載入 RAG 系統...")
    
    # 1. 載入 Embedding
    embedding_model = get_embedding_model()
    
    # 2. 載入 FAISS 向量庫
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"找不到向量庫 {INDEX_PATH}，請先執行 retriever.py")
        
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # 3. 載入父文檔庫 (LocalFileStore + Encoder) <--- 關鍵修改
    fs = LocalFileStore(DOCSTORE_PATH)
    docstore = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
    
    # 4. 重建 Retriever
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 6} 
    )
    
    # 5. 設定 LLM (Ollama - LLaMA 3)
    llm = ChatOllama(
        model="llama3", 
        temperature=0.1, 
    )
    
    # 6. 設定 Prompt Template (強化版：強制中文 + 思維鏈)
    template = """你是一個專業的金融分析師，請協助回答使用者的問題。

    【嚴格遵守規則】
    1. **語言限制**：除非專有名詞，否則**所有回答必須使用繁體中文**。禁止使用英文作答。
    2. **表格閱讀策略**：
       - 若問題涉及「篩選條件」（如：持股 > 20%），請務必**掃描表格的每一列**，不要只看前幾行。
       - 請找出**所有**符合條件的項目，不要遺漏。
       - 若表格中的數字有括號（如 (0.36)），代表負數或減少。
    3. **數據精確性**：回答中的數字必須與文件內容完全一致。
    4. **無答案處理**：若文件中找不到資訊，請直接回答「根據現有文件無法回答」。

    【參考文件片段】
    {context}

    【使用者問題】
    {question}

    【你的分析與回答】(請用繁體中文)："""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 7. 建立 QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

if __name__ == "__main__":
    qa_chain = load_system()
    
    print("\n✅ RAG 系統已就緒！請輸入關於臺銀年報的問題 (輸入 'exit' 離開)")
    print("-" * 50)
    
    while True:
        query = input("\n請輸入問題: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        print("正在思考中...")
        result = qa_chain.invoke({"query": query})
        
        print("\n🤖 回答:")
        print(result['result'])
        
        print("\n📄 參考來源片段:")
        for doc in result['source_documents']:
            print(f"- ...{doc.page_content[:50]}...")