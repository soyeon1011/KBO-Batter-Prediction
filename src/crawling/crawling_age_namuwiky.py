import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time, random, re, sys

print("🚀 [결자해지 봇] 나무위키에서 입단연도를 정확하게 복구합니다!\n")

file_path = '/Users/soyeon/Documents/Capstone/data/KBO_Master_Data_2021_2025.csv'
df = pd.read_csv(file_path)

# 1. 안전장치: 입단연도 관련 컬럼이 없으면 생성
if '입단연도' not in df.columns: df['입단연도'] = None
if '데뷔연차' not in df.columns: df['데뷔연차'] = None

# 2. 엑셀에서 가짜/빈 데이터 색출 (연도 컬럼 기준)
min_seasons = df.groupby(['이름', '팀명'])['연도'].min().to_dict()

def is_bad_data(row):
    name, team = row['이름'], row['팀명']
    b_year, e_year = row['출생연도'], row['입단연도']
    earliest = min_seasons.get((name, team), 2026)
    
    if pd.isnull(e_year): return True # 비어있으면 불량
    if e_year > earliest or e_year < 1980: return True # 입단이 데뷔보다 늦거나 너무 옛날이면 불량
    if pd.notnull(b_year) and e_year <= b_year + 15: return True # 중학생 입단은 불량
    return False

bad_mask = df.apply(is_bad_data, axis=1)
targets = df[bad_mask][['이름', '팀명', '출생연도']].drop_duplicates().values.tolist()

print(f"📊 복구 대상 선수: {len(targets)}명 (약 360여 명)\n")
if not targets: sys.exit("🎉 모든 입단연도가 완벽합니다!")

options = Options()
driver = webdriver.Chrome(options=options)

try:
    for i, (name, team, b_year) in enumerate(targets, 1):
        print(f"🔍 [{i}/{len(targets)}] {team} {name} 나무위키 탐색 중...")
        earliest_season = min_seasons.get((name, team), 2026)
        e_year = None
        
        # 이름(야구선수) -> 이름(팀명) -> 이름 순으로 검색
        for url in [f"https://namu.wiki/w/{name}(야구선수)", f"https://namu.wiki/w/{name}({team})", f"https://namu.wiki/w/{name}"]:
            try:
                driver.get(url)
                time.sleep(random.uniform(2.0, 3.5))
                page_text = driver.find_element(By.TAG_NAME, "body").text
                if "문서가 존재하지 않습니다" in page_text: continue
                
                # 입단 연도 추출
                matches = re.findall(r'(?:프로\s*입단|지명\s*순위|데뷔|입단|지명)[^\d]{0,15}(\d{4})', page_text)
                if not matches: matches = re.findall(r'(\d{4})년[^\n]{0,15}(?:프로\s*입단|입단|지명|데뷔)', page_text)
                
                valid_years = [int(y) for y in matches if 1980 <= int(y) <= earliest_season]
                if valid_years:
                    e_year = valid_years[0]
                    print(f"  ✅ {e_year}년 입단 확인")
                    break
            except: continue
                
        if e_year:
            exp_val = 2026 - e_year + 1
            mask = (df['이름'] == name) & (df['팀명'] == team)
            df.loc[mask, '입단연도'] = e_year
            df.loc[mask, '데뷔연차'] = exp_val
            
            # 10명마다 실시간 저장
            if i % 10 == 0:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"   💾 [저장] {i}명까지 백업 완료")

finally:
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    driver.quit()
    print(f"\n🎉 [작업 완료] 모든 입단연도 복구가 끝났습니다!")