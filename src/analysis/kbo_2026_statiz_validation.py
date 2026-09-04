"""
KBO 2026 시즌 실시간 검증 (스탯티즈 데이터 사용)
=================================================
스탯티즈에서 크롤링한 2026 시즌 데이터와
예측값(kbo_2026_predictions.csv)을 비교

입력:
  statiz_advanced_stats_2026.csv  — 스탯티즈 2026 시즌 실제 데이터
  kbo_2026_predictions.csv        — 모델 예측 타율

출력:
  kbo_2026_validation_MMDD.csv   — 예측 vs 실제 비교 결과
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR     = "/Users/soyeon/Documents/Capstone/data"
PRED_FILE    = os.path.join(DATA_DIR, "kbo_2026_predictions.csv")
STATIZ_FILE  = os.path.join(DATA_DIR, "statiz_advanced_stats_2026.csv")
today        = datetime.today().strftime("%m%d")
OUT_FILE     = os.path.join(DATA_DIR, f"kbo_2026_validation_{today}.csv")

MIN_PA = 50  # 최소 타석 기준

# 외국인 선수 제거
REMOVE_PLAYERS = [
    '케이브', '리베라토', '플로리얼', '카디네스', '위즈덤', '스티븐슨', '맥브룸','데이비슨',
    '오스틴', '페라자', '디아즈', '알칸타라', '로하스', '피렐라', '레이예스','카메론', '세베리노'
]


def main():
    print("=" * 60)
    print(f"KBO 2026 시즌 검증 — 스탯티즈 데이터 ({datetime.today().strftime('%Y-%m-%d')})")
    print("=" * 60)

    # ── 예측값 로드 ───────────────────────────────────────────────────────
    pred_df = pd.read_csv(PRED_FILE)
    if '선수명' not in pred_df.columns:
        pred_df = pred_df.rename(columns={pred_df.columns[1]: '선수명'})
    pred_df = pred_df[['선수명', '2026_예측타율']].copy()
    pred_df['선수명'] = pred_df['선수명'].astype(str).str.strip()
    print(f"\n예측 데이터 로드: {len(pred_df)}명")

    # ── 스탯티즈 실제 데이터 로드 ─────────────────────────────────────────
    statiz_df = pd.read_csv(STATIZ_FILE)
    print(f"스탯티즈 데이터 로드: {len(statiz_df)}명")

    # 컬럼 정리
    statiz_df = statiz_df.rename(columns={'Name': '선수명', 'AVG': '실제타율'})
    statiz_df['선수명'] = statiz_df['선수명'].astype(str).str.strip()
    statiz_df['실제타율'] = pd.to_numeric(statiz_df['실제타율'], errors='coerce')
    statiz_df['PA'] = pd.to_numeric(statiz_df['PA'], errors='coerce').fillna(0).astype(int)

    # 외국인 선수 제거
    statiz_df = statiz_df[~statiz_df['선수명'].isin(REMOVE_PLAYERS)].copy()

    # 최소 타석 필터링
    statiz_df = statiz_df[statiz_df['PA'] >= MIN_PA].copy()
    statiz_df = statiz_df.dropna(subset=['실제타율'])
    print(f"{MIN_PA}타석 이상 + 외국인 제거 후: {len(statiz_df)}명")

    # ── 예측 vs 실제 비교 ─────────────────────────────────────────────────
    actual_df = statiz_df[['선수명', '실제타율', 'PA']].copy()
    merged = pd.merge(pred_df, actual_df, on='선수명', how='inner')
    print(f"매칭된 선수: {len(merged)}명")

    if merged.empty:
        print("\n매칭된 선수가 없습니다. 선수명 형식을 확인하세요.")
        print(f"예측 선수명 예시: {pred_df['선수명'].head(5).tolist()}")
        print(f"스탯티즈 선수명 예시: {actual_df['선수명'].head(5).tolist()}")
        return

    merged['오차']    = (merged['2026_예측타율'] - merged['실제타율']).round(4)
    merged['절대오차'] = merged['오차'].abs().round(4)
    merged = merged.sort_values('절대오차').reset_index(drop=True)
    merged.index += 1

    # ── 성능 지표 계산 ────────────────────────────────────────────────────
    mae  = mean_absolute_error(merged['실제타율'], merged['2026_예측타율'])
    rmse = np.sqrt(mean_squared_error(merged['실제타율'], merged['2026_예측타율']))
    r2   = r2_score(merged['실제타율'], merged['2026_예측타율'])

    within_03 = (merged['절대오차'] < 0.03).sum()
    within_05 = (merged['절대오차'] < 0.05).sum()
    total     = len(merged)

    print(f"\n[현재 시점 검증 성능 — {datetime.today().strftime('%Y.%m.%d')} 기준]")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    print(f"\n[예측 정확도]")
    print(f"  3푼 이내: {within_03}/{total}명 ({within_03/total*100:.1f}%)")
    print(f"  5푼 이내: {within_05}/{total}명 ({within_05/total*100:.1f}%)")

    # ── 잘 맞은 / 크게 틀린 선수 출력 ────────────────────────────────────
    print(f"\n[예측 잘 맞은 선수 TOP 10]")
    print(f"{'순위':>4}  {'선수명':<8}  {'예측타율':>8}  {'실제타율':>8}  {'절대오차':>8}  {'PA':>5}")
    print(f"{'─'*52}")
    for rank, row in merged.head(10).iterrows():
        print(f"{rank:>4}  {str(row['선수명']):<8}  "
              f"{row['2026_예측타율']:>8.4f}  "
              f"{row['실제타율']:>8.4f}  "
              f"{row['절대오차']:>8.4f}  "
              f"{int(row['PA']):>5}")

    print(f"\n[예측 크게 틀린 선수 TOP 10]")
    print(f"{'순위':>4}  {'선수명':<8}  {'예측타율':>8}  {'실제타율':>8}  {'오차':>8}  {'PA':>5}")
    print(f"{'─'*52}")
    for rank, row in merged.tail(10).sort_values('절대오차', ascending=False).iterrows():
        print(f"{rank:>4}  {str(row['선수명']):<8}  "
              f"{row['2026_예측타율']:>8.4f}  "
              f"{row['실제타율']:>8.4f}  "
              f"{row['오차']:>+8.4f}  "
              f"{int(row['PA']):>5}")

    # ── 저장 ──────────────────────────────────────────────────────────────
    merged.to_csv(OUT_FILE, index_label='순위', encoding='utf-8-sig')
    print(f"\n저장 완료: {OUT_FILE}")
    print(f"검증 기준일: {datetime.today().strftime('%Y년 %m월 %d일')} ({total}명 비교)")
    print("=" * 60)


if __name__ == '__main__':
    main()