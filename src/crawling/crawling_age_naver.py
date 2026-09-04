import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import random
import re
import sys

print("🚀 [빈칸 복구 봇] 실시간 저장 모드로 실행합니다. (10명마다 자동 저장)\n")

# 파일 경로
file_path = '/Users/soyeon/Documents/Capstone/data/KBO_Master_Data_2021_2025.csv'
df = pd.read_csv(file_path)

# 1. 안전장치: 컬럼이 아예 없는 경우를 대비해 미리 생성
if 'Age_2026' not in df.columns:
    df['Age_2026'] = None
if '출생연도' not in df.columns:
    df['출생연도'] = None

# 2. Age_2026이 비어있는(NaN) 선수들만 추출
missing_mask = df['Age_2026'].isnull()
unique_missing_players = df[missing_mask][['이름', '팀명']].drop_duplicates().values.tolist()

print(f"📊 복구 대상 선수: {len(unique_missing_players)}명\n")

if len(unique_missing_players) == 0:
    print("🎉 빈칸이 없습니다! 모든 데이터가 이미 채워져 있습니다.")
    sys.exit()

options = Options()
# 필요 시 headless 모드를 추가하여 창 없이 실행 가능합니다.
# options.add_argument('--headless') 
driver = webdriver.Chrome(options=options)

try:
    for i, (name, team) in enumerate(unique_missing_players, 1):
        print(f"🔍 [{i}/{len(unique_missing_players)}] {team} {name} 찾는 중...")
        
        search_query = f"KBO {team} {name}"
        b_year = None
        
        try:
            driver.get(f"https://search.naver.com/search.naver?query={search_query}")
            time.sleep(random.uniform(2.0, 3.5)) # 네이버 차단 방지를 위한 여유로운 휴식
            
            # 동명이인 처리 및 페이지 이동 로직
            player_list = driver.find_elements(By.CSS_SELECTOR, ".list_item, .person_item, .info_list")
            if player_list:
                for item in player_list:
                    if team in item.text:
                        link = item.find_element(By.TAG_NAME, "a")
                        driver.execute_script("arguments[0].click();", link)
                        time.sleep(1.5)
                        break
            
            page_text = driver.find_element(By.TAG_NAME, "body").text
            
            # 캡챠 감지 시 대기
            if "자동입력 방지" in page_text or "비정상적인 검색" in page_text:
                print(f"  🛑 네이버 차단 감지! 60초간 일시 정지합니다...")
                time.sleep(60)
                continue
                
            # 생년월일 정규식 추출
            b_match = re.search(r'(?:출생|생년월일)[\s:]*(\d{4})', page_text)
            if not b_match:
                b_match = re.search(r'(\d{4})\.\d{1,2}\.\d{1,2}', page_text)
                
            if b_match:
                b_year = int(b_match.group(1))
                age_val = 2026 - b_year + 1
                
                # 🚨 [실시간 업데이트] 찾자마자 데이터프레임(df)에 즉시 반영
                mask = (df['이름'] == name) & (df['팀명'] == team)
                df.loc[mask, '출생연도'] = b_year
                df.loc[mask, 'Age_2026'] = age_val
                print(f"  ✅ {b_year}년생 확인 (나이: {age_val})")
            else:
                print(f"  ❌ 정보 없음")

        except Exception as e:
            print(f"  ❌ 에러 발생 (패스)")

        # 🚨 [체크포인트] 10명마다 한 번씩 엑셀 파일에 실제로 저장
        if i % 10 == 0:
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"   💾 [저장] {i}명까지의 데이터를 파일에 안전하게 기록했습니다.")

except KeyboardInterrupt:
    print("\n🛑 사용자에 의해 중단되었습니다. 현재까지의 데이터를 저장합니다.")

finally:
    # 프로그램이 종료되거나 중단되어도 마지막 데이터까지 저장
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    driver.quit()
    print(f"\n🎉 [작업 종료] 모든 복구 데이터가 {file_path}에 최종 저장되었습니다.")