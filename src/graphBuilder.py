import os
import re
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 設定路徑
INPUT_MD = "data/processed/bot_report_113_enriched.md"
if not os.path.exists(INPUT_MD):
    INPUT_MD = "data/processed/bot_report_113_clean.md"

# Neo4j 設定
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def clean_cypher_output(text):
    """清洗 LLM 回傳的 Cypher"""
    text = text.replace("```cypher", "").replace("```", "")
    lines = text.split('\n')
    cleaned_lines = []
    
    # 定義允許的開頭關鍵字
    valid_starts = ("MERGE", "CREATE", "MATCH", "WITH", "UNWIND", "SET", "RETURN", "//")
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        
        # 過濾掉包含 'null' 或無意義的空節點建立指令
        if "null" in stripped.lower() or "''" in stripped or '""' in stripped:
            continue

        if stripped.upper().startswith(valid_starts) or "[:" in stripped or "({" in stripped:
            cleaned_lines.append(stripped)
            
    return "\n".join(cleaned_lines)

def get_extraction_chain(llm):
    # 嚴格定義 Schema (白名單)
    schema = """
    【允許的節點類型 (Nodes)】
    - Company {name: string}  // 公司 (例如：臺灣銀行、華南金控)
    - Department {name: string} // 部門 (例如：資安處、信託部)
    - Person {name: string} // 人名 (例如：凌忠嫄、吳佳曉)
    - Risk {name: string} // 風險項目 (例如：信用風險、氣候變遷風險)
    - Project {name: string} // 專案計畫 (例如：數位轉型旗艦計畫)

    【允許的關係類型 (Relationships)】
    - (:Company)-[:INVESTS_IN {ratio: float}]->(:Company)
    - (:Person)-[:MANAGES]->(:Department)
    - (:Person)-[:SERVES_IN]->(:Company)
    - (:Company)-[:HAS_RISK]->(:Risk)
    - (:Department)-[:EXECUTES]->(:Project)
    """

    template = """你是一個 Neo4j Cypher 專家。請從下方的「文本片段」中抽取實體與關係，並轉換為 Cypher 寫入指令。

    【嚴格規則】
    1. **實體統一**：看到「本行」、「臺銀」、「臺灣銀行股份有限公司」時，節點名稱一律使用 "臺灣銀行"。
    2. **忽略雜訊**：
       - 不要為「年份」(113年)、「地點」(台北市)、「金額」(100億) 建立節點。
       - 不要為通用名詞 (如「客戶」、「員工」、「政府」) 建立節點。
       - 只抽取上述 Schema 定義的 5 種節點。
    3. **格式要求**：
       - 只輸出 Cypher 代碼。
       - 使用 MERGE 指令。
       - 數字與百分比請轉為數字格式 (20% -> 0.2)。
    4. **除錯**：若文本中沒有明確的實體關係，請不要輸出任何指令。

    【Schema】
    {schema}

    【文本片段】
    {text}

    【Cypher】:"""

    prompt = PromptTemplate(
        input_variables=["schema", "text"],
        template=template
    )
    
    return prompt | llm

def build_knowledge_graph():
    print("正在連接 Neo4j...")
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD
        )
        print("Neo4j 連接成功！")
    except Exception as e:
        print(f"Neo4j 連接失敗: {e}")
        return

    # 1. 讀取檔案
    print(f"讀取檔案: {INPUT_MD}")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 2. 切分文本 (加大 Chunk size 以減少切斷關係的情況)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )
    chunks = text_splitter.split_text(full_text)
    
    # 測試模式：只跑前 20 個 chunks 看看效果 (避免一次跑完全部又爆掉)
    # 正式跑時請把 [:20] 拿掉
    chunks_to_process = chunks[:] 
    
    print(f"共切分為 {len(chunks)} 個區塊，預計處理 {len(chunks_to_process)} 個...")

    llm = ChatOllama(model="llama3", temperature=0)
    chain = get_extraction_chain(llm)

    # 3. 迴圈處理
    success_count = 0
    for i, chunk in enumerate(tqdm(chunks_to_process)):
        try:
            response = chain.invoke({"schema": "", "text": chunk})
            cypher_query = clean_cypher_output(response.content)
            
            if not cypher_query.strip():
                continue
            
            # 執行寫入
            queries = cypher_query.split(';')
            for q in queries:
                q = q.strip()
                if q:
                    graph.query(q)
            success_count += 1
            
        except Exception as e:
            # print(f"Chunk {i} Error: {e}")
            pass

    print(f"\n✅ 處理完成！成功寫入 {success_count} 個區塊。")
    print("請前往 Neo4j Browser 檢查節點數量是否合理 (應 < 5000)。")

if __name__ == "__main__":
    build_knowledge_graph()