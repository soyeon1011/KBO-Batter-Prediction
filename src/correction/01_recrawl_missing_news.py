"""
[1단계] 기사가 없는 부상 기록만 네이버 뉴스 다시 크롤링 (투수 크롤러 방식)
==========================================================================
입력 : kbo_injury_final_classified_v31_master.csv  (880건, 기사 없는 87~88건 대상)
출력 : kbo_injury_batter_v32_news.csv               (880건 전체, 기사 보완)

검색 순서 (투수 kbo_pitcher_injury_news_crawling.py와 동일)
  1순위: "팀명 선수명 부상", 부상시작일 -2일 ~ +10일, 단순 필터
  2순위: "선수명 부상",      부상시작일 -5일 ~ +14일, 엄격 필터(이름 뒤 40자, 다른 선수 이름 제외)
  3순위: 2순위와 같은 검색,   완화 필터(이름 뒤 40자에 부상 키워드)
"""
import os
import re
import sys
import time
from urllib.parse import quote

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

DATA_DIR    = "/Users/soyeon/Documents/Capstone/data"
INPUT_FILE  = os.path.join(DATA_DIR, "kbo_injury_final_classified_v31_master.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "kbo_injury_batter_v32_news.csv")
NO_NEWS     = ['관련 기사 없음', '검색 결과 없음']

INJURY_KEYWORDS = [
    '부상', '통증', '말소', '이탈', '결장', '수술', '파열', '재활', '손상',
    '후유증', '소견', '골절', '타박상', '염좌', '불편',
    '어깨', '팔꿈치', '손목', '손가락', '손등', '전완', '광배근',
    '허리', '옆구리', '척추', '요추', '디스크',
    '무릎', '발목', '종아리', '햄스트링', '허벅지',
    '맹장', '장염', '감기', '몸살', '가래톳',
]
SURNAMES = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
            '한', '오', '서', '신', '권', '황', '안', '송', '류', '홍',
            '전', '고', '문', '양', '손', '배', '백', '허', '유', '남',
            '심', '원', '노', '하', '곽', '성', '차', '구', '우', '진',
            '엄', '채', '변', '나', '공']
FALSE_POSITIVE_WORDS = ['이탈', '이후', '이전', '이상', '이하', '이내', '이외',
                        '전날', '전반', '전문', '전체', '전부', '안정', '안타',
                        '강도', '강점', '고통', '고려', '구간', '구성']


def simple_filter(txt, name):
    return name in txt and any(k in txt for k in INJURY_KEYWORDS) and 10 < len(txt) < 500


def strict_filter(txt, name, window=40):
    idx = txt.find(name)
    if idx == -1:
        return False
    after = txt[idx + len(name): idx + len(name) + window]
    if not any(k in after for k in INJURY_KEYWORDS):
        return False
    if ',' in after or '，' in after:
        return False
    for m in re.finditer(r'[' + ''.join(SURNAMES) + r'][가-힣]{1,3}', after):
        w = m.group()
        if w not in FALSE_POSITIVE_WORDS and w != name:
            return False
    return 5 < len(txt) < 400


def soft_filter(txt, name, window=40):
    idx = txt.find(name)
    if idx == -1:
        return False
    after = txt[idx + len(name): idx + len(name) + window]
    return any(k in after for k in INJURY_KEYWORDS) and 5 < len(txt) < 400


def search(driver, query, date_range, name, filter_fn):
    s, e = date_range
    url = (f"https://search.naver.com/search.naver?where=news&query={quote(query)}"
           f"&pd=3&ds={s}&de={e}"
           f"&nso=so%3Ar%2Cp%3Afrom{s.replace('.', '')}to{e.replace('.', '')}%2Ca%3Aall")
    driver.get(url)
    time.sleep(2.5)
    driver.execute_script("window.scrollTo(0, 800);")
    found = []
    for el in driver.find_elements(By.CSS_SELECTOR, "a.news_tit, .api_txt_lines, .news_dsc, .sds-comps-text"):
        txt = el.text.strip().replace("\n", " ")
        if filter_fn(txt, name):
            found.append(txt)
    return found


def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        sys.exit()

    news = df['뉴스기사제목'].fillna('').astype(str).str.strip()
    targets = df.index[(news == '') | news.isin(NO_NEWS)].tolist()
    print(f"✅ {len(df)}건 로드 / 기사 없는 {len(targets)}건 재크롤링")

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    found_cnt = 0
    try:
        for i, idx in enumerate(targets):
            row = df.loc[idx]
            name = str(row['선수']).split('(')[0].strip()
            team = str(row.get('팀', '')).strip()
            d = pd.to_datetime(str(row['부상시작일']).replace('.', '-'))
            narrow = ((d - pd.Timedelta(days=2)).strftime("%Y.%m.%d"), (d + pd.Timedelta(days=10)).strftime("%Y.%m.%d"))
            wide = ((d - pd.Timedelta(days=5)).strftime("%Y.%m.%d"), (d + pd.Timedelta(days=14)).strftime("%Y.%m.%d"))

            print(f"[{i + 1:3d}/{len(targets)}] {name} ({d.date()})", end="  ", flush=True)
            found = search(driver, f"{team} {name} 부상", narrow, name, simple_filter)
            if not found:
                found = search(driver, f"{name} 부상", wide, name, strict_filter)
            if not found:
                found = search(driver, f"{name} 부상", wide, name, soft_filter)

            if found:
                combined = " || ".join(list(dict.fromkeys(found))[:3])
                df.at[idx, '뉴스기사제목'] = combined
                df.at[idx, '뉴스기사unable'] = None
                found_cnt += 1
                print(f"✅ {combined[:50]}...")
            else:
                df.at[idx, '뉴스기사제목'] = '관련 기사 없음'
                df.at[idx, '뉴스기사unable'] = '관련 기사 없음'
                print("❌ 기사 없음")
    except KeyboardInterrupt:
        print("\n🛑 중단 - 저장 중...")
    finally:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        driver.quit()
        print(f"\n💾 저장: {OUTPUT_FILE}  (새로 찾은 기사 {found_cnt}건 / 대상 {len(targets)}건)")


if __name__ == '__main__':
    main()
