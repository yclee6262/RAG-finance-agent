import os
import nest_asyncio
from llama_parse import LlamaParse
from langchain.schema import Document
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

nest_asyncio.apply()
load_dotenv()

def remove_irrelevant_pages(input_pdf_path, output_pdf_path, cutoff_page):
    """
    物理移除 PDF 中不需要的頁面（如分行清單），以節省 Token。
    針對臺銀113年年報，分行資訊通常在最後章節。
    """
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    
    total_pages = len(reader.pages)
    print(f"原始文件共 {total_pages} 頁。")
    
    # 如果 PDF 總頁數少於 cutoff，就全保 (避免切錯檔)
    if total_pages < cutoff_page:
        cutoff_page = total_pages

    for i in range(cutoff_page):
        writer.add_page(reader.pages[i])

    with open(output_pdf_path, "wb") as f:
        writer.write(f)
    
    print(f"已移除分行清單，新文件共 {cutoff_page} 頁。")
    return output_pdf_path

def parse_pdf_to_markdown(pdf_path, cutoff_page, output_path=None):
    """
    使用 LlamaParse 將 PDF 解析為 Markdown 格式，保留表格結構。
    """
    temp_pdf_path = pdf_path.replace(".pdf", "_trimmed.pdf")
    remove_irrelevant_pages(pdf_path, temp_pdf_path, cutoff_page)

    print(f"正在解析: {pdf_path} ... (這可能需要幾分鐘)")
    
    # 設定 LlamaParse
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",  # 關鍵：轉為 Markdown
        # language="zh",           # 設定為繁體中文
        verbose=True,
        num_workers=4            # 加速處理
    )

    # 執行解析
    documents = parser.load_data(pdf_path)
    
    # 將結果合併為一個字串
    full_text = "\n\n".join([doc.text for doc in documents])
    
    # 如果有指定輸出路徑，存成 .md 檔案 (避免每次都要重跑 API)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"解析完成！已儲存至: {output_path}")
    
    # 轉換為 LangChain 的 Document 格式回傳
    return [Document(page_content=full_text, metadata={"source": pdf_path})]

if __name__ == "__main__":
    # 測試用：直接執行此檔案時跑這段
    pdf_file = "data/raw/臺灣銀行113年年報.pdf"  # 請確認檔名與路徑
    md_file = "data/processed/bot_report_113.md"
    cutoff_page = 112
    
    if os.path.exists(pdf_file):
        parse_pdf_to_markdown(pdf_file, cutoff_page, md_file)
    else:
        print(f"找不到檔案: {pdf_file}，請確認檔案已放入 data/raw/")