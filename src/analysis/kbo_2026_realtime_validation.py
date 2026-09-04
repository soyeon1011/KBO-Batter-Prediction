"""
KBO 2026 시즌 실시간 검증
==========================
현재 2026 KBO 시즌 타율을 KBO 공식 사이트에서 크롤링하여
예측값(kbo_2026_predictions.csv)과 비교

출력:
  kbo_2026_validation_MMDD.csv  — 예측 vs 실제 비교 결과
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = "/Users/soyeon/Documents/Capstone/data"
PRED_FILE = os.path.join(DATA_DIR, "kbo_2026_predictions.csv")
today     = datetime.today().strftime("%m%d")
OUT_FILE  = os.path.join(DATA_DIR, f"kbo_2026_validation_{today}.csv")

MIN_PA = 30  # 최소 타석 기준


def crawl_kbo_2026():
    """KBO 공식 사이트에서 2026 시즌 타자 성적 크롤링"""

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    wait = WebDriverWait(driver, 15)

    # 2026 시즌 타자 기본 기록 URL (seasonId 직접 지정)
    url = ("https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx"
           "?leId=1&srId=0&seasonId=2026&teamId=&posId=1&isHitter=1")
    print(f"  URL: {url}")
    driver.get(url)
    time.sleep(3)

    all_rows = []
    seen_names = set()  # 중복 선수 감지용

    def parse_table(driver):
        """현재 페이지 테이블 파싱"""
        from io import StringIO
        html = driver.page_source
        tables = pd.read_html(StringIO(html))
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any('선수' in c or '이름' in c for c in cols) and \
               any('타율' in c or 'AVG' in c for c in cols):
                return t
        return None

    # 1페이지 수집
    print(f"  페이지 1 크롤링 중...")
    df_page = parse_table(driver)
    if df_page is None:
        print("  테이블 파싱 실패")
        driver.quit()
        return pd.DataFrame()

    all_rows.append(df_page)
    print(f"  {len(df_page)}명 수집")

    # 페이지 번호 목록 확인 후 순서대로 클릭
    page_num = 2
    while True:
        try:
            # 숫자 페이지 링크 찾기
            page_links = driver.find_elements(By.CSS_SELECTOR, "div.paging a, .paginate a, ul.paginate_button a")

            if not page_links:
                # 대안: 텍스트가 숫자인 링크 찾기
                all_links = driver.find_elements(By.TAG_NAME, "a")
                page_links = [a for a in all_links if a.text.strip() == str(page_num)]

            target = None
            for link in page_links:
                if link.text.strip() == str(page_num):
                    target = link
                    break

            if target is None:
                # "다음" 버튼 시도
                all_links = driver.find_elements(By.TAG_NAME, "a")
                for link in all_links:
                    if link.text.strip() in ['다음', '>', '▶']:
                        target = link
                        break

            if target is None:
                print(f"  페이지 {page_num} 링크 없음 → 종료")
                break

            print(f"  페이지 {page_num} 크롤링 중...")
            driver.execute_script("arguments[0].click();", target)
            time.sleep(2)

            df_page = parse_table(driver)
            if df_page is None:
                break

            # 중복 체크 (같은 페이지 반복 감지)
            clean = clean_columns(df_page.copy())
            if not clean.empty:
                first_name = clean['선수명'].iloc[0] if '선수명' in clean.columns else ""
                if first_name in seen_names:
                    print(f"  중복 페이지 감지 → 종료")
                    break
                seen_names.update(clean['선수명'].tolist())

            all_rows.append(df_page)
            print(f"  {len(df_page)}명 수집")
            page_num += 1

        except Exception as e:
            print(f"  페이지 {page_num} 오류: {e}")
            break

    driver.quit()

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    return result


def clean_columns(df):
    """컬럼명 정리 및 필요 컬럼 추출"""
    col_map = {}
    for c in df.columns:
        cs = str(c)
        if '선수' in cs or '이름' in cs:
            col_map[c] = '선수명'
        elif '타율' in cs or cs == 'AVG':
            col_map[c] = '실제타율'
        elif '타석' in cs or cs == 'PA':
            col_map[c] = 'PA'
        elif '팀' in cs:
            col_map[c] = '팀'

    df = df.rename(columns=col_map)

    needed = [c for c in ['선수명', '실제타율', 'PA'] if c in df.columns]
    if len(needed) < 2:
        print(f"필요한 컬럼 없음. 현재 컬럼: {list(df.columns)}")
        return pd.DataFrame()

    df = df[needed].copy()
    df['실제타율'] = pd.to_numeric(df['실제타율'], errors='coerce')
    if 'PA' in df.columns:
        df['PA'] = pd.to_numeric(df['PA'], errors='coerce').fillna(0).astype(int)

    df = df.dropna(subset=['선수명', '실제타율'])
    df['선수명'] = df['선수명'].astype(str).str.strip()
    return df


def main():
    print("=" * 60)
    print(f"KBO 2026 시즌 실시간 검증  ({datetime.today().strftime('%Y-%m-%d')})")
    print("=" * 60)

    # ── 예측값 로드 ───────────────────────────────────────────────────────
    pred_df = pd.read_csv(PRED_FILE)
    if '선수명' not in pred_df.columns:
        pred_df = pred_df.rename(columns={pred_df.columns[1]: '선수명'})
    pred_df = pred_df[['선수명', '2026_예측타율']].copy()
    pred_df['선수명'] = pred_df['선수명'].astype(str).str.strip()
    print(f"\n예측 데이터 로드: {len(pred_df)}명")

    # ── 실제 성적 크롤링 ──────────────────────────────────────────────────
    print("\nKBO 공식 사이트 크롤링 시작...")
    raw_df = crawl_kbo_2026()

    if raw_df.empty:
        print("\n크롤링 실패. 브라우저를 직접 열어서 확인이 필요합니다.")
        return

    actual_df = clean_columns(raw_df)
    if actual_df.empty:
        print("\n데이터 파싱 실패.")
        return

    print(f"크롤링 완료: {len(actual_df)}명")

    # 최소 타석 필터링
    if 'PA' in actual_df.columns:
        actual_df = actual_df[actual_df['PA'] >= MIN_PA].copy()
        print(f"최소 {MIN_PA}타석 이상 필터링 후: {len(actual_df)}명")

    # ── 예측 vs 실제 비교 ─────────────────────────────────────────────────
    merged = pd.merge(pred_df, actual_df, on='선수명', how='inner')
    print(f"\n매칭된 선수: {len(merged)}명")

    if merged.empty:
        print("매칭된 선수가 없습니다. 선수명 형식을 확인하세요.")
        print(f"예측 데이터 선수명 예시: {pred_df['선수명'].head(5).tolist()}")
        print(f"실제 데이터 선수명 예시: {actual_df['선수명'].head(5).tolist()}")
        return

    merged['오차']    = (merged['2026_예측타율'] - merged['실제타율']).round(4)
    merged['절대오차'] = merged['오차'].abs().round(4)
    merged = merged.sort_values('절대오차').reset_index(drop=True)
    merged.index += 1

    # ── 성능 지표 ─────────────────────────────────────────────────────────
    mae  = mean_absolute_error(merged['실제타율'], merged['2026_예측타율'])
    rmse = np.sqrt(mean_squared_error(merged['실제타율'], merged['2026_예측타율']))
    r2   = r2_score(merged['실제타율'], merged['2026_예측타율'])

    within_03 = (merged['절대오차'] < 0.03).sum()
    within_05 = (merged['절대오차'] < 0.05).sum()
    total     = len(merged)

    print(f"\n[현재 시점 검증 성능]")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    print(f"\n[예측 정확도]")
    print(f"  3푼 이내: {within_03}/{total}명 ({within_03/total*100:.1f}%)")
    print(f"  5푼 이내: {within_05}/{total}명 ({within_05/total*100:.1f}%)")

    # ── 상위/하위 출력 ────────────────────────────────────────────────────
    print(f"\n[예측 잘 맞은 선수 TOP 10]")
    print(f"{'순위':>4}  {'선수명':<8}  {'예측타율':>8}  {'실제타율':>8}  {'절대오차':>8}")
    print(f"{'─'*48}")
    for rank, row in merged.head(10).iterrows():
        print(f"{rank:>4}  {str(row['선수명']):<8}  "
              f"{row['2026_예측타율']:>8.4f}  "
              f"{row['실제타율']:>8.4f}  "
              f"{row['절대오차']:>8.4f}")

    print(f"\n[예측 크게 틀린 선수 TOP 10]")
    print(f"{'순위':>4}  {'선수명':<8}  {'예측타율':>8}  {'실제타율':>8}  {'오차':>8}")
    print(f"{'─'*48}")
    for rank, row in merged.tail(10).sort_values('절대오차', ascending=False).iterrows():
        print(f"{rank:>4}  {str(row['선수명']):<8}  "
              f"{row['2026_예측타율']:>8.4f}  "
              f"{row['실제타율']:>8.4f}  "
              f"{row['오차']:>+8.4f}")

    # ── 저장 ──────────────────────────────────────────────────────────────
    merged.to_csv(OUT_FILE, index_label='순위', encoding='utf-8-sig')
    print(f"\n저장 완료: {OUT_FILE}")
    print(f"검증 기준일: {datetime.today().strftime('%Y년 %m월 %d일')}")
    print("=" * 60)


if __name__ == '__main__':
    main()