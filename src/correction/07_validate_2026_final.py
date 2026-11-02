"""
[7단계] 2026 시즌 최종 타율과 비교 (독립 시험 평가)
=====================================================
모델은 2025시즌까지의 기록만으로 학습·예측했으므로 2026 기록은 한 번도 쓰지 않았다.
따라서 2026 시즌 최종 타율과의 비교는 독립 시험 평가가 된다.

실행 전 준비
  1) 시즌 종료 후 스탯티즈 2026 구단별 HTML을 다시 저장 (data/html/2026_팀명.html, 10개)
  2) statiz_crawling.py (target_year = 2026) 실행 → statiz_advanced_stats_2026.csv 갱신
  3) 05_kbo_methodB_full.py 실행 결과 results_methodB_v2/예측_2026_선수별.csv 가 있어야 함

출력 (results_methodB_v2 폴더)
  검증_2026_최종_선수별.csv   선수별 예측 타율·실제 타율·오차 (+ 기준선 오차)
  표9_2026_독립시험.csv       Ridge vs 기준선: MAE·RMSE·R², Ridge와의 MAE 차이 95% CI
  표10_2026_타석구간별.csv    2026 타석 구간별 오차
  그림4_2026_예측vs실제.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # 없는 글꼴 경고 숨김
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR  = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
OUT_DIR   = os.path.join(DATA_DIR, "results_methodB_v2")
PRED_FILE = os.path.join(OUT_DIR, "예측_2026_선수별.csv")
STATIZ    = os.path.join(DATA_DIR, "statiz_advanced_stats_2026.csv")
B_FILE    = os.path.join(DATA_DIR, "kbo_final_dataset_methodB_v2.csv")
T4_FILE   = os.path.join(OUT_DIR, "표4_연도별순차평가.csv")
MIN_PA    = 50          # 학습 데이터와 같은 기준
N_BOOT    = 5000
TRADED_2026 = ['하주석', '데이비슨']   # 2026 시즌 중 이적 (한화→KIA, NC→키움)


def metrics(y, p):
    return {'MAE': mean_absolute_error(y, p), 'RMSE': np.sqrt(mean_squared_error(y, p)), 'R2': r2_score(y, p)}


def boot_ci(y, pa, pb, seed=0):
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    n = len(y)
    diffs = [(ea - eb)[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (ea - eb).mean(), lo, hi


def main():
    pred = pd.read_csv(PRED_FILE)
    stz = pd.read_csv(STATIZ)
    stz = stz.rename(columns={'Name': '선수명', 'PA': '2026_PA', 'AB': '2026_AB', 'H': '2026_H', 'AVG': '2026_실제타율'})
    stz['선수명'] = stz['선수명'].astype(str).str.strip()

    # 시즌 중 이적한 선수는 두 팀 파일에 모두 나오므로 기록을 합친다 (타율 = 안타 / 타수)
    traded = stz[stz['선수명'].isin(TRADED_2026)]
    if len(traded):
        merged = traded.groupby('선수명', as_index=False)[['2026_PA', '2026_AB', '2026_H']].sum()
        merged['2026_실제타율'] = (merged['2026_H'] / merged['2026_AB']).round(3)
        stz = pd.concat([stz[~stz['선수명'].isin(TRADED_2026)], merged], ignore_index=True)
        print(f"시즌 중 이적 → 두 팀 기록 합산: {sorted(merged['선수명'])}")

    # 그래도 같은 이름이 2명 이상이면(동명이인) 비교에서 제외
    dup = stz['선수명'][stz['선수명'].duplicated(keep=False)].unique()
    if len(dup):
        print(f"동명이인으로 제외: {sorted(dup)}")
    stz = stz[~stz['선수명'].isin(dup)]

    df = pred.merge(stz[['선수명', '2026_PA', '2026_AB', '2026_H', '2026_실제타율']], on='선수명', how='inner')
    n_pred = len(pred)
    df = df[df['2026_PA'] >= MIN_PA].copy()
    print(f"예측 {n_pred}명 중 2026 기록 {MIN_PA}타석 이상 {len(df)}명 비교")

    y = df['2026_실제타율'].values
    p = df['2026_예측타율'].values

    # 기준선 (예측 시점 = 2025시즌 종료 후에 알 수 있는 정보만 사용)
    base = pd.read_csv(B_FILE)
    b25 = base[base['연도'] == 2025]
    lg25 = b25['H_KBO'].sum() / b25['AB_KBO'].sum()
    k = 150
    if os.path.exists(T4_FILE):
        k = int(pd.read_csv(T4_FILE).iloc[-1]['평균회귀 k'])
    h25 = (df['2025_타율'] * df['2025_AB']).round()
    baselines = {
        '전년도 타율': df['2025_타율'].values,
        '리그 평균(2025)': np.full(len(df), lg25),
        f'평균회귀(k={k})': ((h25 + k * lg25) / (df['2025_AB'] + k)).values,
    }

    rows = [{'예측 방법': 'Ridge', **metrics(y, p), 'Ridge−기준선 MAE': np.nan, 'CI 하한': np.nan, 'CI 상한': np.nan}]
    for name, b in baselines.items():
        d, lo, hi = boot_ci(y, p, b)
        rows.append({'예측 방법': name, **metrics(y, b), 'Ridge−기준선 MAE': d, 'CI 하한': lo, 'CI 상한': hi})
    t9 = pd.DataFrame(rows).round(4)
    print("\n[표9] 2026 독립 시험 평가"); print(t9.to_string(index=False))
    t9.to_csv(os.path.join(OUT_DIR, '표9_2026_독립시험.csv'), index=False, encoding='utf-8-sig')

    err = np.abs(y - p)
    print(f"\n절대오차 중앙값 {np.median(err):.4f} / 75% {np.percentile(err, 75):.4f} / 90% {np.percentile(err, 90):.4f}"
          f" / 평균 편향(예측−실제) {np.mean(p - y):+.4f}")

    df['오차'] = (df['2026_예측타율'] - df['2026_실제타율']).round(4)
    df['절대오차'] = df['오차'].abs()
    df['타석 구간'] = pd.cut(df['2026_PA'], [MIN_PA - 1, 150, 300, 450, 800],
                         labels=[f'{MIN_PA}~150', '151~300', '301~450', '451 이상'])
    t10 = df.groupby('타석 구간', observed=True)['절대오차'].agg(['size', 'mean']).rename(
        columns={'size': '선수 수', 'mean': 'MAE'}).reset_index().round(4)
    print("\n[표10] 타석 구간별 오차"); print(t10.to_string(index=False))
    t10.to_csv(os.path.join(OUT_DIR, '표10_2026_타석구간별.csv'), index=False, encoding='utf-8-sig')

    for name, b in baselines.items():
        df[f'{name}_절대오차'] = np.abs(y - b).round(4)
    out = df.sort_values('절대오차').reset_index(drop=True)
    out.index += 1
    out.to_csv(os.path.join(OUT_DIR, '검증_2026_최종_선수별.csv'), index_label='순위', encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.scatter(p, y, s=14, color='#4a6fa5', alpha=.75, edgecolor='none')
    lo = np.floor(min(p.min(), y.min()) * 50) / 50
    hi = np.ceil(max(p.max(), y.max()) * 50) / 50
    ax.plot([lo, hi], [lo, hi], color='#888', lw=1, ls='--', label='예측 = 실제')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel('2026 예측 타율 (Ridge)'); ax.set_ylabel('2026 시즌 최종 타율')
    ax.set_title(f'2026 독립 시험: 예측 vs 실제 (n={len(df)}, MAE={err.mean():.4f})', fontsize=10)
    ax.legend(loc='lower right', fontsize=8, frameon=False); ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '그림4_2026_예측vs실제.png')); plt.close()
    print(f"\n저장 → {OUT_DIR}")


if __name__ == '__main__':
    main()