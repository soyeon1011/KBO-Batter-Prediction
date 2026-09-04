import pandas as pd
import time
import io
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

# 🌟 경로 설정 (수정된 파일명 반영)
# kbo_futures_game.py 상단 파일명 확인
INPUT_FILE = "/Users/soyeon/Documents/Capstone/data/kbo_injury_batters_only_final.csv"
OUTPUT_FILE = "/Users/soyeon/Documents/Capstone/data/kbo_injury_futures_final_result.csv"

# 1. 데이터 로드 (원본 파일은 읽기만 수행)
try:
    df = pd.read_csv(INPUT_FILE)
    # 결장 기간 100일 이하인 선수들만 우선 수색 대상으로 선정
    suspects_df = df[(df['실제결장일수'] > 0) & (df['실제결장일수'] <= 100)].copy()
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}"); sys.exit()

print(f"🕵️‍♂️ [정밀 수사] 타자 {len(suspects_df)}명을 수색합니다. (브라우저 확인 가능)")

# 2. 셀레니움 설정 (창 띄움)
options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

futures_detail_list = []

try:
    for index, row in suspects_df.iterrows():
        player_name = row['선수']
        team = row['팀']
        # 괄호 제거 로직 ('김도영(내야수)' -> '김도영')
        clean_name = str(player_name).split('(')[0].strip()
        start_date = pd.to_datetime(row['부상시작일'])
        end_date = pd.to_datetime(row['실제복귀일'])
        year = str(start_date.year)
        
        print(f"🔍 [{index+1}/{len(suspects_df)}] {team} {clean_name} 확인 중...", end=" ", flush=True)
        
        try:
            # Step 1: 선수 고유 ID 확보
            driver.get(f"https://www.koreabaseball.com/Player/Search.aspx?searchWord={clean_name}")
            time.sleep(1.5)
            
            player_id = ""
            if "Search.aspx" in driver.current_url:
                rows = driver.find_elements(By.CSS_SELECTOR, ".tEx tBody tr")
                for r in rows:
                    if team in r.text:
                        player_id = r.find_element(By.TAG_NAME, "a").get_attribute("href").split("playerId=")[1]
                        break
            else:
                player_id = driver.current_url.split("playerId=")[1].split("&")[0]
            
            if not player_id:
                print("❌ 검색불가"); futures_detail_list.append("검색불가"); continue

            # Step 2: 2군 일자별 기록실 순간이동
            driver.get(f"https://www.koreabaseball.com/Futures/Player/HitterDaily.aspx?playerId={player_id}")
            time.sleep(1.5)

            # Step 3: 연도 선택
            try:
                select = Select(driver.find_element(By.ID, "cphContents_cphContents_cphContents_ddlYear"))
                select.select_by_value(year)
                time.sleep(2)
            except: pass

            # Step 4: 테이블 분석 및 상세 날짜 추출
            html_source = driver.page_source
            if "데이터가 없습니다" in html_source:
                print("✅ 통과"); futures_detail_list.append("X"); continue

            all_tables = pd.read_html(io.StringIO(html_source))
            valid_tables = []
            for t in all_tables:
                if '상대' in t.columns:
                    new_cols = list(t.columns); new_cols[0] = '날짜'; t.columns = new_cols
                    valid_tables.append(t)
            
            if not valid_tables:
                print("✅ 통과"); futures_detail_list.append("X"); continue

            # 데이터 병합 및 날짜 필터링
            df_futures = pd.concat(valid_tables, ignore_index=True)
            df_futures['날짜_정제'] = year + "." + df_futures['날짜'].astype(str)
            df_futures['날짜_정제'] = pd.to_datetime(df_futures['날짜_정제'], format='%Y.%m.%d', errors='coerce')
            
            played = df_futures[(df_futures['날짜_정제'] >= start_date) & (df_futures['날짜_정제'] <= end_date)]
            
            if not played.empty:
                # 결과 포맷: "n경기 (MM.DD, MM.DD)"
                dates_str = ", ".join(played['날짜_정제'].dt.strftime('%m.%d').tolist())
                detail = f"{len(played)}경기 ({dates_str})"
                print(f"🚨 적발! {detail}")
                futures_detail_list.append(detail)
            else:
                print("✅ 통과"); futures_detail_list.append("X")

        except Exception as e:
            print("⚠️ 에러"); futures_detail_list.append("에러")

except KeyboardInterrupt:
    print("\n🛑 사용자에 의해 조사가 중단되었습니다.")

finally:
    # 🌟 [안전 저장 프로토콜] 기존 원본 데이터 보호
    if len(futures_detail_list) > 0:
        actual_len = len(futures_detail_list)
        final_df = suspects_df.iloc[:actual_len].copy()
        final_df['2군출장상세'] = futures_detail_list
        
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n💾 수사 완료! 결과는 새로운 파일에 저장되었습니다: {OUTPUT_FILE}")
    else:
        print("\n🚨 유효한 수집 데이터가 없어 파일을 생성하지 않았습니다.")
    
    driver.quit()