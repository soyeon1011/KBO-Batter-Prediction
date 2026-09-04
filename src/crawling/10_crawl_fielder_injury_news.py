import pandas as pd
import time
import sys
import os
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 1. 경로 설정
DATA_DIR = "/Users/soyeon/Documents/Capstone/data"
INPUT_FILE = os.path.join(DATA_DIR, "kbo_injury_source_v3_active_only.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "kbo_injury_source_v4_crawled.csv")

# 2. 데이터 로딩
try:
    df = pd.read_csv(INPUT_FILE)
    # 날짜 형식 통일 (하이픈으로 변경 후 datetime 변환)
    df['부상시작일'] = pd.to_datetime(df['부상시작일'].astype(str).str.replace('.', '-', regex=False))
    print(f"✅ 수사 대상 {len(df)}건 로드 완료")
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    sys.exit()

# 3. 크롬 설정 (소연님의 원래 잘 되던 설정)
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_argument("--start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 수사 키워드
INJURY_KEYWORDS = ['손상', '부상', '옆구리', '파열', '수술', '통증', '재활', '말소', '이탈', '후유증', '결장', '소견', '교통사고', '골절', '붓기', '타박상', '종아리', '햄스트링', '허리']

# 4. 크롤링 대상 필터링 (기사 제목이 비어있는 행만)
mask = df['뉴스기사제목'].isna() | (df['뉴스기사제목'] == '') | (df['뉴스기사제목'] == 'To_Be_Crawled')
target_indices = df[mask].index

print(f"🔍 전체 중 신규 수색 대상: {len(target_indices)}건")

try:
    for i, idx in enumerate(target_indices):
        row = df.loc[idx]
        player_name = str(row['선수']).split('(')[0].strip()
        team = row['팀']
        injury_date = row['부상시작일']
        
        # 검색 기간 설정 (부상일 전후로 정밀 타격)
        start_dt = (injury_date - pd.Timedelta(days=2)).strftime("%Y.%m.%d")
        end_dt = (injury_date + pd.Timedelta(days=10)).strftime("%Y.%m.%d")
        start_no_dot = start_dt.replace(".", "")
        end_no_dot = end_dt.replace(".", "")

        # 네이버 뉴스 상세 검색 URL 조합
        query = f"{team} {player_name} 부상"
        search_url = (f"https://search.naver.com/search.naver?where=news&query={quote(query)}"
                      f"&pd=3&ds={start_dt}&de={end_dt}"
                      f"&nso=so%3Ar%2Cp%3Afrom{start_no_dot}to{end_no_dot}%2Ca%3Aall")

        print(f"\n🔍 [{i+1}/{len(target_indices)}] {player_name} ({injury_date.date()}) 수색 중...", end=" ", flush=True)
        driver.get(search_url)
        time.sleep(2.5) # 로딩 대기
        
        driver.execute_script("window.scrollTo(0, 800);")

        found_info = []

        # 소연님이 지정해주신 4가지 핵심 태그 수집
        targets = driver.find_elements(By.CSS_SELECTOR, "a.news_tit, .api_txt_lines, .news_dsc, .sds-comps-text")
        
        for t in targets:
            txt = t.text.strip().replace("\n", " ")
            # 이름 검증 + 부상 키워드 검증
            if player_name in txt and any(k in txt for k in INJURY_KEYWORDS):
                if 10 < len(txt) < 500: # 노이즈 제거
                    found_info.append(txt)

        if found_info:
            # 중복 제거 후 최대 3개 문장 병합
            combined = " || ".join(list(set(found_info))[:3])
            df.at[idx, '뉴스기사제목'] = combined
            print(f"✅", end="")
        else:
            df.at[idx, '뉴스기사unable'] = "관련 기사 없음"
            print("❌", end="")

        # 중간 저장 (혹시 멈출 경우 대비, 10건마다 저장)
        if (i + 1) % 10 == 0:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

except KeyboardInterrupt:
    print("\n🛑 사용자에 의해 중단되었습니다.")

finally:
    # 최종 결과 저장
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n💾 수사 종료! 파일 저장 완료: {OUTPUT_FILE}")
    driver.quit()