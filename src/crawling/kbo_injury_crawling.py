from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from io import StringIO
import sys
import re

options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

print("KBO 선수 이동 현황 페이지 접속 중...")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.koreabaseball.com/Player/Trade.aspx")
time.sleep(3)

# 1. 연도 세팅
# target_year = "2021"
# target_year = "2022"
# target_year = "2023"
# target_year = "2024"
target_year = "2025"
try:
    # 무식하지만 가장 확실한 방법: 페이지의 모든 select(드롭다운) 태그를 싹 다 찾음
    dropdowns = driver.find_elements(By.TAG_NAME, "select")
    
    # 0번째(가장 첫 번째) 드롭다운이 무조건 연도 선택 상자임!
    year_select = Select(dropdowns[0])
    
    # value 값이 다를 수 있으니, 우리 눈에 보이는 글자("2024")로 강제 선택
    year_select.select_by_visible_text(target_year)
    print(f"✅ {target_year}년도로 변경 완료!")
    time.sleep(4)
    
except Exception as e:
    print(f"🚨 연도 선택 실패: {e}")

all_data = []
page = 1

# 2. 페이지 순회하며 데이터 긁기
print(f"\n[{target_year}년] 데이터 수집 시작...")
while True:
    html = driver.page_source
    if "<table" in html:
        tables = pd.read_html(StringIO(html))
        if len(tables) > 0:
            df = tables[0]
            df.columns = df.columns.str.replace('\n', ' ')
            all_data.append(df)
            print(f"  ✅ {page}페이지 수집 완료")

    try:
            if page % 10 != 0:
                # 1~9, 11~19 페이지: 숫자 그대로 클릭
                next_btn = driver.find_element(By.LINK_TEXT, str(page + 1))
            else:
                # 🌟 10, 20, 30 페이지: KBO 개발자의 alt 오타를 무시하고 
                # 무조건 진짜 '다음' 버튼인 class="pg_next"를 정밀 타격!!!
                next_btn = driver.find_element(By.CSS_SELECTOR, ".pg_next")
                print(f"  🚀 {page}페이지 도달: 진짜 다음 블록(.pg_next) 클릭 성공!")

            # 화면 중앙 스크롤 & 자바스크립트 강제 클릭
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", next_btn)
            
            # 표가 껌뻑! 하고 데이터가 완전히 바뀔 때까지 넉넉히 대기
            time.sleep(4) 
            
            page += 1

    except Exception as e:
            print(f"  👉 수집 종료! (총 {page}페이지 진행)")
            # print(f"  🚨 에러 원인: {e}") 
            break
    

driver.quit()

if not all_data:
    print("🚨 수집된 데이터가 없습니다.")
    sys.exit()

# 3. 데이터 병합
final_df = pd.concat(all_data, ignore_index=True)

# 4. 🌟 도메인 지식 반영: 부상뿐만 아니라 '등록', '말소' 기록까지 전부 포함해서 필터링
# 그래야 나중에 '부상(말소) -> 등록' 짝을 맞춰서 실제 결장 일수를 계산할 수 있음!
roster_df = final_df[final_df['항목'].str.contains('부상|치료|재활|등록|말소', na=False, regex=True)].copy()

# 5. 비고란 숫자 추출 (유지)
def extract_days(text):
    if pd.isna(text):
        return 0
    numbers = re.findall(r'\d+', str(text))
    return int(numbers[0]) if numbers else 0

roster_df['부상_일수_텍스트'] = roster_df['비고'].apply(extract_days)

# 6. 파일 저장
save_path = f"/Users/soyeon/Documents/Capstone/data/kbo_injury_list_{target_year}.csv"
roster_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n🎉 완벽하게 성공! 총 [{len(roster_df)}건]의 1군 등록/말소 데이터가 저장되었습니다.")