import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def inspect_data():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    with driver.session() as session:
        print("\n=== 1. 檢查公司節點名稱 (確認是「台」還是「臺」) ===")
        # 抓出所有連出 INVESTS_IN 關係的節點名稱
        result = session.run("""
            MATCH (n)-[:INVESTS_IN]->() 
            RETURN DISTINCT n.name as name 
            LIMIT 5
        """)
        for record in result:
            print(f"Investor Name: '{record['name']}'")

        print("\n=== 2. 檢查持股比例格式 (確認是 0.2 還是 20) ===")
        # 抓出前 5 筆投資關係的數值
        result = session.run("""
            MATCH (n)-[r:INVESTS_IN]->(m) 
            RETURN n.name, r.ratio, m.name 
            LIMIT 5
        """)
        for record in result:
            print(f"{record['n.name']} --[ratio: {record['r.ratio']}]--> {record['m.name']}")

    driver.close()

if __name__ == "__main__":
    inspect_data()