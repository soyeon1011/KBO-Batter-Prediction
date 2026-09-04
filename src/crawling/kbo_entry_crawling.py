from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from io import StringIO
import sys

options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

print("KBO 전체 등록 현황 페이지 접속 중...")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.koreabaseball.com/Player/RegisterAll.aspx")

# target_year = "2021"
# target_year = "2022"
# target_year = "2023"
# target_year = "2024"
target_year = "2025"

print("\n🚨 [수동 조작 타임] 🚨")
print(f"브라우저가 열렸습니다! 달력을 눌러서 크롤링을 시작할 날짜(예: {target_year}년 3월 23일)를 세팅해 주세요.")
print("30초 뒤에 파이썬이 바통을 터치받아 크롤링을 시작합니다...")
time.sleep(30)

print("\n🚀 파이썬 자동 수집 시작! (맨 밑바닥 등/말소 표만 타겟팅)")

all_data = []
empty_days_count = 0 

# 약 180일(시즌 전체) 루프
for _ in range(200): 
    try:
        time.sleep(2) 
        
        # 1. 현재 달력 날짜 가져오기 (예: 2024.03.29)
        date_element = driver.find_element(By.CLASS_NAME, "date-txt")
        current_date = date_element.text[:10] # '2024.03.29(금)' -> '2024.03.29'
        
        daily_dfs = []
        
        # 2. 소연님이 찾아낸 '1군 등록 현황' 표 긁기
        try:
            reg_html = driver.find_element(By.CLASS_NAME, "fistStatus").get_attribute('outerHTML')
            reg_df = pd.read_html(StringIO(reg_html))[0]
            reg_df['상태'] = '등록'
            daily_dfs.append(reg_df)
        except:
            pass # 등록된 선수가 없는 날은 패스
            
        # 3. 소연님이 찾아낸 '1군 말소 현황' 표 긁기
        try:
            cancel_html = driver.find_element(By.CLASS_NAME, "fistCancelStatus").get_attribute('outerHTML')
            cancel_df = pd.read_html(StringIO(cancel_html))[0]
            cancel_df['상태'] = '말소'
            daily_dfs.append(cancel_df)
        except:
            pass # 말소된 선수가 없는 날은 패스

        # 4. 데이터 합치기
        if daily_dfs:
            day_df = pd.concat(daily_dfs, ignore_index=True)
            day_df['일자'] = current_date.replace('.', '-') # 2024-03-29 형태로 변환
            all_data.append(day_df)
            print(f"✅ {current_date} - 수집 완료 (총 {len(day_df)}명 변동)")
            empty_days_count = 0
        else:
            print(f"  - {current_date} - 변동 없음")
            empty_days_count += 1
            
        if empty_days_count >= 15:
            print("🛑 15일 연속 변동이 없어 시즌 종료로 간주합니다.")
            break

        # 5. 소연님이 찾아준 다음(>) 화살표 버튼 클릭!
        next_btn = driver.find_element(By.XPATH, "//img[@alt='다음날짜']")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", next_btn)

    except Exception as e:
        print(f"🚨 크롤링 중단: {e}")
        break

driver.quit()

if not all_data:
    print("🚨 수집된 데이터가 없습니다.")
    sys.exit()

# 최종 데이터 정리
final_df = pd.concat(all_data, ignore_index=True)
final_df = final_df[['일자', '팀', '선수', '포지션', '상태']] # 컬럼 순서 예쁘게 정리

save_path = f"/Users/soyeon/Documents/Capstone/data/kbo_entry_history_{target_year}.csv"
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n🎉 완벽 성공! 총 [{len(final_df)}건]의 등/말소 로그가 '{save_path}'에 저장되었습니다.")