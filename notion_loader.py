import os
import pandas as pd
from dotenv import load_dotenv
from notion_client import Client

# ㄴ.env 파일에 저장된 환경변수 로드
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# 노션 클라이언트 초기화
notion = Client(auth = NOTION_TOKEN)

def fetch_notion_data(database_id):
    print("노션 데이터베이스에서 데이터를 가져오는 중...")
    results = []
    has_more = True
    start_cursor = None
    
    while has_more:
        response = notion.databases.query(
            database_id = database_id,
            start_cursor = start_cursor
        )
        results.extend(response.get("results"))
        has_more = response.get("has_more")
        start_cursor = response.get("next_cursor")
        
    print(f"Successfully fetched {len(results)} rows of data!")
    return results

def parse_notion_properties(results):
    parsed_data = []
    
    for row in results:
        properties = row.get("properties", {})
        row_data = {}
        
        for prop_name, prop_content in properties.items():
            prop_type = prop_content.get("type")
            
            # 1. 텍스트 / 타이틀 타입 추출
            if prop_type in ["title", "rich_text"]:
                text_list = prop_content.get(prop_type, [])
                row_data[prop_name] = text_list[0].get("plain_text", "") if text_list else ""
                
            # 2. 숫자 타입 추출
            elif prop_type == "number":
                row_data[prop_name] = prop_content.get("number", None)
                
            # 3. 선택형(Select) 타입 추출
            elif prop_type == "select":
                select_obj = prop_content.get("select")
                row_data[prop_name] = select_obj.get("name", "") if select_obj else ""
                
            # 4. 다중 선택(Multi-select) 타입 추출
            elif prop_type == "multi_select":
                ms_list = prop_content.get("multi_select", [])
                row_data[prop_name] = ", ".join([item.get("name", "") for item in ms_list])
                
            # 5. 날짜 타입 추출
            elif prop_type == "date":
                date_obj = prop_content.get("date")
                row_data[prop_name] = date_obj.get("start", "") if date_obj else ""
                
            # 6. 체크박스 타입 추출
            elif prop_type == "checkbox":
                row_data[prop_name] = prop_content.get("checkbox", False)
                
        parsed_data.append(row_data)
        
    df = pd.DataFrame(parsed_data)
    return df

if __name__ == "__main__":
    raw_results = fetch_notion_data(DATABASE_ID)
    df = parse_notion_properties(raw_results)
    
    print("\n--- 정제된 데이터프레임 구조 확인 ---")
    print(df.head())
    
    df.to_csv("notion_brain_dump_raw.csv", index = False, encoding = "utf-8-sig")
    print("\n'notion_brain_dump_raw.csv' 파일로 저장 완료!")