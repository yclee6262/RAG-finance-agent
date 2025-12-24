import os
import pickle
import torch
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# =================設定區=================
INDEX_PATH = "data/faiss_index"
DOCSTORE_PATH = "data/doc_store"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# =================模型載入=================
def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_llm():
    return ChatOllama(model="llama3", temperature=0)

# =================向量檢索 (Vector Retrieval)=================
def get_vector_retriever():
    print("載入向量資料庫 (FAISS)...")
    embedding_model = get_embedding_model()
    
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    fs = LocalFileStore(DOCSTORE_PATH)
    docstore = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3} 
    )

# =================圖譜檢索 (Graph Retrieval)=================
def get_graph_chain():
    print("載入知識圖譜 (Neo4j)...")
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD
    )
    
    # 【修正】所有的 Cypher 語法括號都要變成 {{ }}
    cypher_template = """
    你是一個 Neo4j Cypher 專家。請根據以下的 Schema 定義，將使用者的問題轉換為 Cypher 查詢。
    
    【Schema 定義 (請嚴格遵守)】
    Node Labels: 
      - Company (公司)
      - Person (人物)
      - Department (部門)
      - Risk (風險)
      - Project (計畫)
      
    Relationship Types: 
      - INVESTS_IN {{ratio: float, amount: int}}
      - MANAGES
      - HAS_RISK
    
    【嚴格規則】
    1. **Label 必須是英文**：請務必使用 `Company`，**絕對禁止**使用 `公司`、`Firm` 等中文或同義詞。
    2. **實體名稱維持中文**：查詢內容維持中文，例如 {{name: "臺灣銀行"}}。
    3. **關係屬性**：持股比例屬性為 `ratio` (格式為小數，0.2 代表 20%)。
    4. **語法範例**：
       - 錯誤：MATCH (n:公司 {{name: "臺灣銀行"}})...
       - 正確：MATCH (n:Company {{name: "臺灣銀行"}})-[r:INVESTS_IN]->(m:Company) WHERE r.ratio > 0.2 RETURN m.name, r.ratio
    5. 只輸出 Cypher 代碼，不要有 Markdown 標記。
    
    問題：{question}
    Cypher："""
    
    PROMPT = PromptTemplate(input_variables=["question"], template=cypher_template)
    
    return GraphCypherQAChain.from_llm(
        get_llm(),
        graph=graph,
        verbose=True,
        cypher_prompt=PROMPT,
        allow_dangerous_requests=True,
        return_direct=True 
    )

# =================混合檢索引擎 (Hybrid Engine)=================
class HybridRAG:
    def __init__(self):
        self.llm = get_llm()
        self.vector_retriever = get_vector_retriever()
        self.graph_chain = get_graph_chain()
        
    def query(self, user_query):
        print(f"\n🚀 正在處理問題: {user_query}")
        
        # 1. 平行執行兩路檢索
        # Path A: Vector Search (找文本脈絡)
        print("   [1/3] 執行向量檢索...")
        vector_docs = self.vector_retriever.get_relevant_documents(user_query)
        vector_context = "\n".join([d.page_content for d in vector_docs])
        
        # Path B: Graph Search (找精確數據)
        print("   [2/3] 執行圖譜檢索...")
        graph_context = ""
        try:
            # 這裡我們用 try-except，因為有些問題圖譜查不到 (例如：公司願景)
            # 如果圖譜查詢報錯或查無資料，就不參考圖譜
            graph_result = self.graph_chain.invoke(user_query)
            graph_data = graph_result['result']
            if graph_data:
                graph_context = f"【圖譜資料庫數據】: {str(graph_data)}"
        except Exception as e:
            print(f"   (圖譜檢索跳過: {e})")
            
        # 2. 最終融合生成 (Synthesis)
        print("   [3/3] 融合資訊並生成回答...")
        
        final_prompt = f"""
        你是一個金融分析專家。請根據以下兩個來源的資訊回答問題。
        
        來源 1 - 向量文件 (包含詳細敘述)：
        {vector_context}
        
        來源 2 - 知識圖譜 (包含精確數值與關係)：
        {graph_context}
        
        【回答規則】
        1. **優先信任知識圖譜的數值**：如果問題涉及「持股比例」、「金額」、「人名」，且圖譜有資料，請以圖譜為準。
        2. **使用向量文件補充細節**：利用來源 1 的內容來解釋背景或補充圖譜沒提到的資訊。
        3. 請使用繁體中文回答。
        
        使用者問題：{user_query}
        
        回答：
        """
        
        response = self.llm.invoke(final_prompt)
        return response.content

# =================主程式=================
if __name__ == "__main__":
    app = HybridRAG()
    
    # 測試題庫
    test_questions = [
        "請列出臺灣銀行持股比例超過 20% 的轉投資事業。", # (圖譜強項)
        "請說明本行的資通安全風險管理架構。",           # (向量強項)
        "113年溫室氣體減量的目標是什麼？"               # (向量強項)
    ]
    
    for q in test_questions:
        print("="*60)
        answer = app.query(q)
        print(f"\n🤖 最終回答:\n{answer}\n")