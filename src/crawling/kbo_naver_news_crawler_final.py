import pandas as pd
import time
import sys
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 1. 경로 설정
INPUT_FILE = "/Users/soyeon/Documents/Capstone/data/kbo_injury_with_pre_stats.csv"
OUTPUT_FILE = "/Users/soyeon/Documents/Capstone/data/kbo_injury_final_with_news_final_fixed.csv"

# 2. 데이터 로딩
try:
    df = pd.read_csv(INPUT_FILE)
    df['부상시작일'] = pd.to_datetime(df['부상시작일'].astype(str).str.replace('.', '-', regex=False))
    print(f"✅ 수사 대상 {len(df)}건 로드 완료")
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}"); sys.exit()

# 3. 크롬 설정 (원래 잘 되던 설정 그대로)
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_argument("--start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 수사 키워드
INJURY_KEYWORDS = ['손상', '부상', '옆구리', '파열', '수술', '통증', '재활', '말소', '이탈', '후유증', '결장', '소견', '교통사고', '골절', '붓기', '타박상', '종아리', '햄스트링', '허리']

news_results = []

try:
    for index, row in df.iterrows():
        player_name = str(row['선수']).split('(')[0].strip()
        team = row['팀']
        injury_date = row['부상시작일']
        
        # 검색 기간 설정
        start_dt = (injury_date - pd.Timedelta(days=2)).strftime("%Y.%m.%d")
        end_dt = (injury_date + pd.Timedelta(days=10)).strftime("%Y.%m.%d")
        start_no_dot = start_dt.replace(".", "")
        end_no_dot = end_dt.replace(".", "")

        # 검색 쿼리
        query = f"{team} {player_name} 부상"
        search_url = (f"https://search.naver.com/search.naver?where=news&query={quote(query)}"
                      f"&pd=3&ds={start_dt}&de={end_dt}"
                      f"&nso=so%3Ar%2Cp%3Afrom{start_no_dot}to{end_no_dot}%2Ca%3Aall")

        print(f"\n🔍 [{index+1}/{len(df)}] {player_name} 수색 중...", end=" ", flush=True)
        driver.get(search_url)
        time.sleep(3) # 로딩 대기
        
        driver.execute_script("window.scrollTo(0, 1000);")

        found_info = []

        # 🌟 소연님이 개발자 도구로 알려주신 그 태그들로만 싹 긁어옵니다.
        targets = driver.find_elements(By.CSS_SELECTOR, "a.news_tit, .api_txt_lines, .news_dsc, .sds-comps-text")
        
        for t in targets:
            txt = t.text.strip().replace("\n", " ")
            # 이름이 있고, 부상 관련 키워드가 하나라도 있으면 수집
            if player_name in txt and any(k in txt for k in INJURY_KEYWORDS):
                if 10 < len(txt) < 500: # 너무 짧거나 긴 메뉴 텍스트 제외
                    found_info.append(txt)

        if found_info:
            # 중복 제거 후 최대 3개 문장 병합 (상세 내용 확보)
            combined = " || ".join(list(set(found_info))[:3])
            print(f"✅", end="")
            news_results.append(combined)
        else:
            print("❌", end="")
            news_results.append("관련 기사 없음")

except KeyboardInterrupt:
    print("\n🛑 중단")

finally:
    if news_results:
        final_df = df.iloc[:len(news_results)].copy()
        final_df['뉴스기사제목'] = news_results
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장 완료: {OUTPUT_FILE}")
    driver.quit()