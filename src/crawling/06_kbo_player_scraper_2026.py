import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 1. 크롬 드라이버 설정 (창이 직접 보이도록 설정)
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
wait = WebDriverWait(driver, 10)

def scrape_kbo_active_fielders_2026():
    try:
        url = "https://www.koreabaseball.com/Player/Search.aspx"
        driver.get(url)
        
        all_fielders = []
        target_positions = ['내야수', '외야수', '포수']

        # 2. 팀 드롭다운 메뉴에서 전체 팀 리스트 확보 (고양, 상무 포함)
        team_select_element = wait.until(EC.presence_of_element_located((By.ID, "cphContents_cphContents_cphContents_ddlTeam")))
        team_select = Select(team_select_element)
        team_names = [opt.text for opt in team_select.options if opt.text != "팀 선택"]

        print(f"🚀 2026년 가용 야수 명단 수집 시작 (전 구단 + 고양/상무 포함)...")

        for team in team_names:
            print(f"⚾ [{team}] 수집 시작...")
            
            # 구단 선택
            Select(driver.find_element(By.ID, "cphContents_cphContents_cphContents_ddlTeam")).select_by_visible_text(team)
            time.sleep(2) 

            # 3. 1페이지부터 5페이지까지 클릭하며 데이터 수집
            for page in range(1, 6):
                try:
                    # 페이지 번호 클릭 (ID 규칙: btnNo1, btnNo2...)
                    page_btn_id = f"cphContents_cphContents_cphContents_ucPager_btnNo{page}"
                    page_btn = driver.find_element(By.ID, page_btn_id)
                    page_btn.click()
                    time.sleep(2) # 데이터 로딩 대기
                    
                    # 4. 주신 HTML 구조(Index 3: 포지션)에 맞춰 데이터 추출
                    rows = driver.find_elements(By.CSS_SELECTOR, ".tEx tbody tr")
                    for row in rows:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        if len(cols) > 3:
                            name = cols[1].text
                            pos = cols[3].text
                            
                            # 야수만 필터링
                            if pos in target_positions:
                                all_fielders.append({
                                    '팀': team, # 고양, 상무 등 원래 소속팀명 그대로 저장
                                    '선수': name,
                                    '포지션': pos
                                })
                except:
                    # 페이지가 5페이지보다 적으면(예: 상무는 보통 1-2페이지) 루프 탈출
                    break
        
        # 5. 데이터프레임 변환 및 중복 제거
        # (여러 페이지나 구단에 중복 노출되는 경우 방지)
        df = pd.DataFrame(all_fielders)
        df = df.drop_duplicates(subset=['선수', '포지션'], keep='first')
        
        # 저장 경로
        output_path = "/Users/soyeon/Documents/Capstone/data/kbo_registration_2026.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("-" * 50)
        print(f"✨ 수집 완료! 총 {len(df)}명의 야수 명단이 확보되었습니다.")
        print(f"📁 저장 파일: {output_path}")
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    finally:
        print("💡 수집이 끝났습니다. 창은 수동으로 닫아주세요.")

if __name__ == "__main__":
    scrape_kbo_active_fielders_2026()