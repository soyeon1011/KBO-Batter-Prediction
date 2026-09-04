"""
집계 방법 A/B/C 성능 비교 스크립트
====================================
세 방법 각각 동일한 모델로 학습 후 RMSE / MAE / R² 비교
→ 성능 제일 좋은 방법을 최종 데이터셋으로 선택

시간 기반 분할 (시계열 데이터이므로 random split 금지):
  Train    : 연도 ≤ 2023  (2021→2022, 2022→2023, 2023→2024 타율 예측)
  Validate : 연도 == 2024 (2024→2025 타율 예측)
  Predict  : 연도 == 2025 (AVG_KBO_next=NaN, 2026 예측 대상)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ xgboost 없음 → pip install xgboost 로 설치 가능")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠ lightgbm 없음 → pip install lightgbm 로 설치 가능")

DATA_DIR = "/Users/soyeon/Documents/Capstone/data"
FILES = {
    'A (합산+최대심각도)': os.path.join(DATA_DIR, "kbo_final_dataset_methodA.csv"),
    'B (최악 1건)':       os.path.join(DATA_DIR, "kbo_final_dataset_methodB.csv"),
    'C (합산+평균심각도)': os.path.join(DATA_DIR, "kbo_final_dataset_methodC.csv"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature 컬럼 목록 (필요에 따라 추가/제거)
# ─────────────────────────────────────────────────────────────────────────────
STAT_FEATURES = [
    'AVG_KBO', 'PA_KBO', 'AB_KBO', 'H_KBO', 'HR_KBO', 'RBI_KBO',
    'OBP', 'SLG', 'OPS', 'wRC+', 'WAR', 'BABIP', 'ISO', 'wOBA',
    'BB/K', '데뷔연차', 'Age_2026', 'Park_Factor',
]
INJURY_FEATURES_A_C = [
    '부상발생여부', '부상횟수', '총결장일수', '최대심각도', '최고신뢰도점수',
]
INJURY_FEATURES_B = [
    '부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수',
]
CATEGORICAL = ['대표부상분류']
TARGET = 'AVG_KBO_next'


def prepare_xy(df, injury_feat):
    """feature / target 분리 + 범주형 인코딩"""
    all_feat = STAT_FEATURES + injury_feat + CATEGORICAL
    # 존재하는 컬럼만 사용
    use_feat = [c for c in all_feat if c in df.columns]

    X = df[use_feat].copy()
    y = df[TARGET]

    # 범주형 인코딩
    for col in CATEGORICAL:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # 수치형 결측 → 0
    X = X.fillna(0)
    return X, y


def evaluate(name, df, injury_feat):
    """Train / Validate 분할 후 3가지 모델 평가"""
    # 학습용만 (target 있는 행)
    df_train_full = df[df[TARGET].notna()].copy()

    train = df_train_full[df_train_full['연도'] <= 2023]
    valid = df_train_full[df_train_full['연도'] == 2024]

    X_train, y_train = prepare_xy(train, injury_feat)
    X_valid, y_valid = prepare_xy(valid, injury_feat)

    models = {
        'RandomForest':     RandomForestRegressor(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Ridge':            Ridge(alpha=1.0),
        'Lasso':            Lasso(alpha=0.001),
        'ElasticNet':       ElasticNet(alpha=0.001, l1_ratio=0.5),
        'SVR':              Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1.0))]),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                          max_depth=3, random_state=42,
                                          verbosity=0)
    if HAS_LGB:
        models['LightGBM'] = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                            max_depth=3, random_state=42,
                                            verbose=-1)

    print(f"\n{'─'*55}")
    print(f"방법 {name}  (train: {len(train)}행, valid: {len(valid)}행)")
    print(f"{'─'*55}")

    results = {}
    for mname, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        mae  = mean_absolute_error(y_valid, pred)
        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        r2   = r2_score(y_valid, pred)

        results[mname] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
        print(f"  {mname:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    return results


def main():
    print("=" * 55)
    print("집계 방법 A / B / C 성능 비교")
    print("=" * 55)

    all_results = {}

    for method_name, filepath in FILES.items():
        df = pd.read_csv(filepath)

        # 방법 B는 컬럼명이 다름
        if 'B' in method_name:
            injury_feat = INJURY_FEATURES_B
        else:
            injury_feat = INJURY_FEATURES_A_C

        res = evaluate(method_name, df, injury_feat)
        all_results[method_name] = res

    # ── 최종 요약 ──
    print(f"\n{'='*65}")
    print("최종 요약 — 방법별 최고 성능 모델")
    print(f"{'='*65}")
    print(f"{'방법':<22} {'최고모델':<16} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"{'─'*65}")

    best_combo_method = None
    best_combo_model  = None
    best_combo_rmse   = float('inf')

    for method_name, res in all_results.items():
        # 각 방법에서 RMSE 가장 낮은 모델 찾기
        best_m = min(res, key=lambda m: res[m]['RMSE'])
        r = res[best_m]
        print(f"{method_name:<22} {best_m:<16} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R²']:>8.4f}")
        if r['RMSE'] < best_combo_rmse:
            best_combo_rmse   = r['RMSE']
            best_combo_method = method_name
            best_combo_model  = best_m

    print(f"\n✅ 추천: {best_combo_method}  +  {best_combo_model}  (RMSE={best_combo_rmse:.4f})")
    print("=" * 65)


if __name__ == '__main__':
    main()