"""
[8단계] 2026 예측 타율 vs 2026 정규시즌 최종 타율 — 선수별 비교표
====================================================================
입력 : kbo_2026_predictions_methodB_v2.csv  (06단계 결과)
       statiz_advanced_stats_2026.csv       (시즌 종료 후 갱신한 스탯티즈 기록)
출력 : kbo_2026_validation_final.csv
       순위, 선수명, 2026_예측타율, 실제타율, PA, 오차, 절대오차  (절대오차 작은 순)
"""
import os
import pandas as pd

DATA_DIR  = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
PRED_FILE = os.path.join(DATA_DIR, "kbo_2026_predictions_methodB_v2.csv")
STATIZ    = os.path.join(DATA_DIR, "statiz_advanced_stats_2026.csv")
OUT_FILE  = os.path.join(DATA_DIR, "kbo_2026_validation_final.csv")
MIN_PA    = 50
TRADED_2026 = ['하주석', '데이비슨']   # 시즌 중 이적 → 두 팀 기록 합산

pred = pd.read_csv(PRED_FILE)[['선수명', '2026_예측타율']]

stz = pd.read_csv(STATIZ)
stz['Name'] = stz['Name'].astype(str).str.strip()
# 이적 선수: 두 팀 기록 합치기
trd = stz[stz['Name'].isin(TRADED_2026)].groupby('Name', as_index=False)[['PA', 'AB', 'H']].sum()
trd['AVG'] = trd['H'] / trd['AB']
stz = pd.concat([stz[~stz['Name'].isin(TRADED_2026)][['Name', 'PA', 'AB', 'H', 'AVG']], trd], ignore_index=True)
# 그래도 이름이 겹치면 동명이인 → 제외
dup = stz['Name'][stz['Name'].duplicated(keep=False)].unique()
if len(dup):
    print(f"동명이인으로 제외: {sorted(dup)}")
stz = stz[~stz['Name'].isin(dup)]

df = pred.merge(stz.rename(columns={'Name': '선수명'}), on='선수명', how='inner')
df = df[df['PA'] >= MIN_PA].copy()

df['실제타율'] = df['AVG'].round(3)
df['오차'] = (df['2026_예측타율'] - df['실제타율']).round(4)
df['절대오차'] = df['오차'].abs()
out = df[['선수명', '2026_예측타율', '실제타율', 'PA', '오차', '절대오차']] \
    .sort_values('절대오차').reset_index(drop=True)
out.index += 1
out.to_csv(OUT_FILE, index_label='순위', encoding='utf-8-sig')

print(f"예측 {len(pred)}명 중 2026 기록 {MIN_PA}타석 이상 {len(out)}명 비교")
print(f"MAE {out['절대오차'].mean():.4f}")
print(f"저장 → {OUT_FILE}")
print(out.head(10).to_string())