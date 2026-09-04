"""
KBO Ridge 계수 분석
===================
에이징커브 / 구장효과 / 부상영향 / 성적 변수
각각 타율에 얼마나 영향 미치는지 수치로 확인
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR   = "/Users/soyeon/Documents/Capstone/data"
INPUT_FILE = os.path.join(DATA_DIR, "kbo_final_dataset_methodC.csv")

STAT_FEATURES = [
    'AVG_KBO', 'PA_KBO', 'AB_KBO', 'H_KBO', 'HR_KBO', 'RBI_KBO',
    'OBP', 'SLG', 'OPS', 'wRC+', 'WAR', 'BABIP', 'ISO', 'wOBA',
    'BB/K', '데뷔연차', 'Age_2026', 'Park_Factor',
]
INJURY_FEATURES = [
    '부상발생여부', '부상횟수', '총결장일수', '평균심각도', '최고신뢰도점수',
]
CATEGORICAL = ['대표부상분류']
TARGET = 'AVG_KBO_next'


def prepare_xy(df):
    all_feat = STAT_FEATURES + INJURY_FEATURES + CATEGORICAL
    use_feat = [c for c in all_feat if c in df.columns]
    X = df[use_feat].copy()
    if '대표부상분류' in X.columns:
        le = LabelEncoder()
        X['대표부상분류'] = le.fit_transform(X['대표부상분류'].astype(str))
    X = X.fillna(0)
    y = df[TARGET]
    return X, y, use_feat


def main():
    df = pd.read_csv(INPUT_FILE)
    df_train = df[df[TARGET].notna()].copy()

    X, y, feat_names = prepare_xy(df_train)

    # ── 표준화 후 Ridge 학습 (계수 비교 위해 스케일 맞춤) ──────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # ── 계수 정리 ─────────────────────────────────────────────────────────
    coef_df = pd.DataFrame({
        'feature': feat_names,
        'coefficient': model.coef_,
        'abs_coef': np.abs(model.coef_),
        'direction': ['↑ 타율 상승' if c > 0 else '↓ 타율 하락' for c in model.coef_]
    }).sort_values('abs_coef', ascending=False)

    print("=" * 60)
    print("Ridge 계수 분석 — 표준화 기준 (영향력 크기 순)")
    print("=" * 60)
    print(f"{'순위':>4}  {'변수명':<18}  {'계수':>8}  {'방향'}")
    print("─" * 55)
    for i, row in coef_df.iterrows():
        rank = coef_df.index.get_loc(i) + 1
        print(f"{rank:>4}  {row['feature']:<18}  {row['coefficient']:>+8.4f}  {row['direction']}")

    # ── 핵심 3가지 요소 요약 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("핵심 요소별 영향력 요약")
    print(f"{'='*60}")

    # 에이징커브
    age_coef = coef_df[coef_df['feature'] == 'Age_2026']['coefficient'].values
    debut_coef = coef_df[coef_df['feature'] == '데뷔연차']['coefficient'].values
    print(f"\n[에이징커브]")
    if len(age_coef) > 0:
        direction = "나이 들수록 타율 하락" if age_coef[0] < 0 else "나이 들수록 타율 상승"
        print(f"  Age_2026  계수: {age_coef[0]:+.4f}  →  {direction}")
    if len(debut_coef) > 0:
        direction = "연차 쌓일수록 타율 하락" if debut_coef[0] < 0 else "연차 쌓일수록 타율 상승"
        print(f"  데뷔연차  계수: {debut_coef[0]:+.4f}  →  {direction}")

    # 구장 효과
    park_coef = coef_df[coef_df['feature'] == 'Park_Factor']['coefficient'].values
    print(f"\n[구장 효과]")
    if len(park_coef) > 0:
        direction = "타자 친화 구장일수록 타율 상승" if park_coef[0] > 0 else "타자 친화 구장일수록 타율 하락"
        print(f"  Park_Factor  계수: {park_coef[0]:+.4f}  →  {direction}")

    # 부상 영향
    inj_features = ['부상발생여부', '총결장일수', '평균심각도', '부상횟수']
    print(f"\n[부상 영향]")
    for feat in inj_features:
        c = coef_df[coef_df['feature'] == feat]['coefficient'].values
        if len(c) > 0:
            direction = "타율 상승" if c[0] > 0 else "타율 하락"
            print(f"  {feat:<12}  계수: {c[0]:+.4f}  →  {direction}")

    # ── CSV 저장 ─────────────────────────────────────────────────────────
    OUT = os.path.join(DATA_DIR, "kbo_ridge_coefficients.csv")
    coef_df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f"\n계수 전체 저장: {OUT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
