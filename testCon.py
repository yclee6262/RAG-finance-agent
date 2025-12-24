import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

print(f"1. 嘗試連線至: {URI}")
print(f"2. 使用者: {USER}")
print(f"3. 密碼: {PASSWORD[:2]}***{PASSWORD[-2:] if PASSWORD else ''}") # 遮蔽密碼

try:
    # 嘗試建立驅動程式 (不驗證憑證，排除 SSL 問題)
    # 這裡我們顯式設定 trust=TRUST_ALL_CERTIFICATES 以防萬一
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    # 嘗試驗證連線
    print("4. 正在驗證連線...")
    driver.verify_connectivity()
    
    print("✅ 連線成功！Neo4j 運作正常。")
    driver.close()

except Exception as e:
    print("\n❌ 連線失敗！詳細錯誤如下：")
    print("-" * 30)
    print(e)
    print("-" * 30)
    
    # 錯誤分析建議
    error_msg = str(e)
    if "Connection refused" in error_msg:
        print("💡 原因分析：找不到伺服器。")
        print("   - 請確認 Neo4j Desktop 的 Instance 是否呈現「🟢 Active/Running」狀態？")
        print("   - 如果你在 WSL 跑 Python，但 Neo4j 在 Windows，請看下方的「WSL 解決方案」。")
    elif "Authentication failure" in error_msg or "The client is unauthorized" in error_msg:
        print("💡 原因分析：密碼錯誤。")
        print("   - 請確認 .env 中的密碼是否與 Neo4j Desktop 設定的一致。")
        print("   - 你可以在 Neo4j Desktop 重設密碼 (Instance 右邊的三個點 -> Reset password)。")
    elif "SSL" in error_msg or "certificate" in error_msg:
        print("💡 原因分析：加密/憑證問題。")
        print("   - 請嘗試將 .env 的 URI 改為：bolt://127.0.0.1:7687")