import pandas as pd
import time
import io
import sys
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

# 1. 경로 설정
DATA_DIR = "/Users/soyeon/Documents/Capstone/data"
INPUT_FILE = os.path.join(DATA_DIR, "kbo_injury_source_v4_crawled.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "kbo_injury_source_v5_futures_final.csv")

# 2. 데이터 로드
try:
    df = pd.read_csv(INPUT_FILE)
    # 날짜 형식 변환
    df['부상시작일'] = pd.to_datetime(df['부상시작일'])
    df['실제복귀일'] = pd.to_datetime(df['실제복귀일'])
    print(f"🕵️‍♂️ [2군 정밀 수사] 총 {len(df)}건의 부상 이력을 검증합니다.")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}"); sys.exit()

# 3. 셀레니움 설정 (창 띄움 모드)
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

futures_results = []

try:
    for index, row in df.iterrows():
        player_name = str(row['선수']).split('(')[0].strip()
        team = row['팀']
        start_date = row['부상시작일']
        end_date = row['실제복귀일']
        year = str(start_date.year)
        
        print(f"🔍 [{index+1}/{len(df)}] {team} {player_name} ({year}) 확인 중...", end=" ", flush=True)
        
        try:
            # Step 1: 선수 고유 ID 확보
            driver.get(f"https://www.koreabaseball.com/Player/Search.aspx?searchWord={player_name}")
            time.sleep(1.2)
            
            player_id = ""
            rows = driver.find_elements(By.CSS_SELECTOR, ".tEx tBody tr")
            for r in rows:
                # 팀명 매칭 (키움/고양 예외 처리 포함)
                if team in r.text or (team == '키움' and '고양' in r.text):
                    player_id = r.find_element(By.TAG_NAME, "a").get_attribute("href").split("playerId=")[1]
                    break
            
            if not player_id:
                print("❌ 검색불가"); futures_results.append("확인불가"); continue

            # Step 2: 2군 타자 일자별 기록실 접속
            driver.get(f"https://www.koreabaseball.com/Futures/Player/HitterDaily.aspx?playerId={player_id}")
            time.sleep(1.2)

            # Step 3: 해당 연도 선택
            try:
                select = Select(driver.find_element(By.ID, "cphContents_cphContents_cphContents_ddlYear"))
                select.select_by_value(year)
                time.sleep(1.5)
            except: pass

            # Step 4: 테이블 분석
            html_source = driver.page_source
            if "데이터가 없습니다" in html_source:
                print("✅ 0경기"); futures_results.append("0경기"); continue

            all_tables = pd.read_html(io.StringIO(html_source))
            valid_table = None
            for t in all_tables:
                if '상대' in t.columns:
                    # 첫 번째 컬럼 이름을 '날짜'로 강제 지정 (데이터 전처리용)
                    new_cols = list(t.columns)
                    new_cols[0] = '날짜'
                    t.columns = new_cols
                    valid_table = t
                    break
            
            if valid_table is None:
                print("✅ 0경기"); futures_results.append("0경기"); continue

            # 날짜 필터링 (결장 기간 내 출장 여부 확인)
            valid_table['날짜_정제'] = year + "." + valid_table['날짜'].astype(str)
            valid_table['날짜_정제'] = pd.to_datetime(valid_table['날짜_정제'], format='%Y.%m.%d', errors='coerce')
            
            played = valid_table[(valid_table['날짜_정제'] >= start_date) & (valid_table['날짜_정제'] <= end_date)]
            
            if not played.empty:
                dates_str = ", ".join(played['날짜_정제'].dt.strftime('%m.%d').tolist())
                detail = f"{len(played)}경기 ({dates_str})"
                print(f"🚨 적발! {detail}")
                futures_results.append(detail)
            else:
                print("✅ 0경기"); futures_results.append("0경기")

        except Exception as e:
            print(f"⚠️ 에러"); futures_results.append("에러")

        # 10건마다 중간 저장
        if (index + 1) % 10 == 0:
            temp_df = df.iloc[:len(futures_results)].copy()
            temp_df['2군출장상세'] = futures_results
            temp_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

except KeyboardInterrupt:
    print("\n🛑 중단되었습니다.")

finally:
    if futures_results:
        df['2군출장상세'] = futures_results
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n💾 수사 완료! 최종 저장: {OUTPUT_FILE}")
    driver.quit()