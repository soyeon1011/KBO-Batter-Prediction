import pandas as pd
import time
import io
import sys
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 🌟 경로 설정
INPUT_FILE = "/Users/soyeon/Documents/Capstone/data/kbo_injury_futures_final_result.csv"
OUTPUT_FILE = "/Users/soyeon/Documents/Capstone/data/kbo_injury_with_pre_stats.csv"

# 1. 데이터 로드
try:
    df = pd.read_csv(INPUT_FILE)
    df['부상시작일'] = pd.to_datetime(df['부상시작일'])
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}"); sys.exit()

print(f"🕵️‍♂️ [성과 수사] {len(df)}건의 부상 전 성적 수집을 시작합니다.")

# 2. 셀레니움 설정
options = webdriver.ChromeOptions()
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

pre_stats_list = []

try:
    for index, row in df.iterrows():
        player_name = str(row['선수']).split('(')[0].strip()
        team = row['팀']
        injury_date = row['부상시작일']
        year = str(injury_date.year)
        
        print(f"🔍 [{index+1}/{len(df)}] {team} {player_name} ({year}년)", end=" ", flush=True)
        
        try:
            # Step 1: 선수 검색
            driver.get(f"https://www.koreabaseball.com/Player/Search.aspx?searchWord={player_name}")
            time.sleep(1.2)
            
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
                print("❌ 검색불가"); pre_stats_list.append(None); continue

            # Step 2: 일자별 성적 접속
            driver.get(f"https://www.koreabaseball.com/Record/Player/HitterDetail/Daily.aspx?playerId={player_id}")
            
            # Step 3: 연도 선택
            try:
                year_select = wait.until(EC.presence_of_element_located((By.ID, "cphContents_cphContents_cphContents_ddlYear")))
                select = Select(year_select)
                if select.first_selected_option.get_attribute("value") != year:
                    select.select_by_value(year)
                    time.sleep(2)
            except: pass

            # Step 4: 테이블 수집 (보내주신 HTML 구조 반영)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.tbl.tt")))
            html_source = driver.page_source
            all_tables = pd.read_html(io.StringIO(html_source))
            
            valid_dfs = []
            for t in all_tables:
                if '상대' in t.columns:
                    # 🌟 [KeyError 해결] 컬럼명을 표준화합니다.
                    # 보내주신 HTML 기준: 0:날짜, 4:AB(타수), 6:H(안타)
                    t.columns = ['날짜', '상대', 'AVG1', 'PA', 'AB', 'R', 'H'] + list(t.columns[7:])
                    valid_dfs.append(t[['날짜', 'AB', 'H']])
            
            if not valid_dfs:
                print("✅ 기록없음"); pre_stats_list.append(0.0); continue

            daily_df = pd.concat(valid_dfs, ignore_index=True)
            
            # 날짜 정제
            daily_df['날짜_정제'] = year + "." + daily_df['날짜'].astype(str)
            daily_df['날짜_정제'] = pd.to_datetime(daily_df['날짜_정제'], format='%Y.%m.%d', errors='coerce')
            
            # 부상 전 경기 필터링
            pre_injury_games = daily_df[daily_df['날짜_정제'] < injury_date].sort_values('날짜_정제', ascending=False)
            
            if len(pre_injury_games) == 0:
                print("✅ 시즌첫부상"); pre_stats_list.append(0.0); continue

            # 🌟 최근 10경기 타율 계산 (AB, H 컬럼 사용)
            last_10 = pre_injury_games.head(10)
            hits = pd.to_numeric(last_10['H'], errors='coerce').sum()
            at_bats = pd.to_numeric(last_10['AB'], errors='coerce').sum()
            
            avg = hits / at_bats if at_bats > 0 else 0.0
            print(f"📊 {len(last_10)}경기 타율: {avg:.3f}")
            pre_stats_list.append(round(avg, 3))

        except Exception as e:
            print(f"⚠️ 에러: {type(e).__name__}"); pre_stats_list.append(None)

except KeyboardInterrupt:
    print("\n🛑 중단되었습니다.")

finally:
    if len(pre_stats_list) > 0:
        actual_len = len(pre_stats_list)
        final_df = df.iloc[:actual_len].copy()
        final_df['직전10경기타율'] = pre_stats_list
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n💾 수집 완료! 결과 저장: {OUTPUT_FILE}")
    
    driver.quit()