import re
import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# 設定檔案路徑
INPUT_MD = "data/processed/bot_report_113.md"
OUTPUT_MD = "data/processed/bot_report_113_enriched.md"

def extract_tables(markdown_text):
    """
    使用 Regex 簡單抓取 Markdown 表格
    (尋找 |---|---| 這種結構的區塊)
    """
    # 這是比較簡單的 regex，抓取連續的表格行
    table_pattern = r'((?:\|.*\|\n)+)'
    tables = re.findall(table_pattern, markdown_text)
    return tables

def summarize_table(table_content, llm):
    """
    將表格內容丟給 LLM 產生自然語言摘要
    """
    prompt = PromptTemplate.from_template(
        """你是一個金融數據分析師。請閱讀以下的 Markdown 表格，並用繁體中文撰寫一段「自然語言摘要」。
        
        【摘要規則】
        1. **精準識別主題**：請根據表格的第一列或標題，準確說明這張表是在列出什麼（例如：「轉投資事業」、「財務績效」等），不要隨意縮小範圍。
        2. **全面掃描數據**：請掃描整張表格的數據欄位。
        3. **關鍵項目列表**：
           - 請找出數值(如持股比例、金額) **最大** 的前 5 個項目。
           - 請找出數值 **超過 20%** 的所有項目。
           - 若有「合計」或「總計」，請特別列出。
        4. **禁止幻覺**：不要編造表格中沒有的趨勢。
        
        請不要輸出 Markdown 格式，直接輸出文字摘要即可。

        表格內容：
        {table}

        摘要："""
    )
    
    chain = prompt | llm
    try:
        response = chain.invoke({"table": table_content})
        return response.content
    except Exception as e:
        print(f"摘要生成失敗: {e}")
        return ""

def process_markdown_with_summaries():
    if not os.path.exists(INPUT_MD):
        print(f"找不到檔案: {INPUT_MD}")
        return

    with open(INPUT_MD, "r", encoding="utf-8") as f:
        content = f.read()

    print("正在偵測表格...")
    tables = extract_tables(content)
    print(f"共發現 {len(tables)} 個表格區域。")

    if len(tables) == 0:
        print("未發現表格，可能是 Regex 沒抓到，或 LlamaParse 格式差異。")
        return

    llm = ChatOllama(model="llama3", temperature=0)
    
    new_content = content
    
    print("正在生成表格摘要 (這需要一點時間)...")
    for table in tqdm(tables):
        # 如果表格太短(可能是誤判)，跳過
        if len(table) < 50: 
            continue
            
        summary = summarize_table(table, llm)
        
        if summary:
            # 將摘要插入在表格的上方，並加上標記
            enriched_block = f"\n\n> **表格摘要**: {summary}\n\n{table}"
            # 替換原文 (這裡用 replace 簡單處理，嚴謹做法應依位置替換)
            new_content = new_content.replace(table, enriched_block)

    # 存檔
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"✅ 處理完成！已產出增強版 Markdown: {OUTPUT_MD}")

if __name__ == "__main__":
    process_markdown_with_summaries()