"""
KBO 2026 타율 예측 — Ridge 최근 연도 가중치 모델
=================================================
방법 C (총결장일수 합산 + 평균심각도) 데이터 사용
모델: Ridge Regression — 최근 연도일수록 높은 가중치 부여

가중치 방식 (지수적):
  2021년 → 1
  2022년 → 2
  2023년 → 4
  2024년 → 8  (2026 예측 시 전체 학습 기준)

출력:
  kbo_2026_predictions_weighted.csv  — 선수별 2026 예측 타율
  kbo_backtest_2025_weighted.csv     — 백테스팅 결과 (기본 Ridge vs 가중치 Ridge)
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/soyeon/Documents/Capstone/data"
INPUT_FILE = os.path.join(DATA_DIR, "kbo_final_dataset_methodC.csv")
OUT_PRED   = os.path.join(DATA_DIR, "kbo_2026_predictions_weighted.csv")
OUT_BT     = os.path.join(DATA_DIR, "kbo_backtest_2025_weighted.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Feature 목록
# ─────────────────────────────────────────────────────────────────────────────
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

# 연도별 가중치 (지수적: 최근일수록 2배씩 증가)
YEAR_WEIGHTS = {
    2021: 1,
    2022: 2,
    2023: 4,
    2024: 8,
    2025: 10,
}


def prepare_xy(df, fit_encoder=True, le=None):
    all_feat = STAT_FEATURES + INJURY_FEATURES + CATEGORICAL
    use_feat = [c for c in all_feat if c in df.columns]
    X = df[use_feat].copy()

    if '대표부상분류' in X.columns:
        if fit_encoder:
            le = LabelEncoder()
            X['대표부상분류'] = le.fit_transform(X['대표부상분류'].astype(str))
        else:
            known = set(le.classes_)
            X['대표부상분류'] = X['대표부상분류'].astype(str).apply(
                lambda v: v if v in known else le.classes_[0]
            )
            X['대표부상분류'] = le.transform(X['대표부상분류'])

    X = X.fillna(0)
    y = df[TARGET] if TARGET in df.columns else None
    return X, y, le


def get_sample_weights(df):
    """연도 컬럼 기준으로 샘플 가중치 생성"""
    return df['연도'].map(YEAR_WEIGHTS).fillna(1).values


def print_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<35}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return r2


def main():
    print("=" * 65)
    print("KBO 2026 타율 예측 — Ridge 최근 연도 가중치 모델")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)
    print(f"\n데이터 로드: {df.shape[0]}행 × {df.shape[1]}열")

    df_labeled = df[df[TARGET].notna()].copy()
    df_predict = df[(df[TARGET].isna()) & (df['연도'] == 2025)].copy()

    print(f"  학습용: {len(df_labeled)}행  /  예측용: {len(df_predict)}행")

    # 연도별 가중치 현황 출력
    print(f"\n[연도별 가중치]")
    for yr, w in sorted(YEAR_WEIGHTS.items()):
        cnt = len(df_labeled[df_labeled['연도'] == yr])
        if cnt > 0:
            print(f"  {yr}년: 가중치 {w:>2}  ({cnt}행)")

    # ── 시계열 분할 ───────────────────────────────────────────────────────
    train_df = df_labeled[df_labeled['연도'] <= 2023]
    valid_df = df_labeled[df_labeled['연도'] == 2024]

    X_tr, y_tr, le = prepare_xy(train_df, fit_encoder=True)
    X_va, y_va, _  = prepare_xy(valid_df, fit_encoder=False, le=le)

    # 학습 데이터 가중치
    w_tr = get_sample_weights(train_df)

    # ── 기본 Ridge (가중치 없음) ──────────────────────────────────────────
    ridge_base = Ridge(alpha=1.0)
    ridge_base.fit(X_tr, y_tr)
    pred_base = ridge_base.predict(X_va)

    # ── 가중치 Ridge ──────────────────────────────────────────────────────
    ridge_weighted = Ridge(alpha=1.0)
    ridge_weighted.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_weighted = ridge_weighted.predict(X_va)

    # ── 성능 비교 ─────────────────────────────────────────────────────────
    print(f"\n[검증 성능 비교 — Valid: 2024년 {len(valid_df)}행]")
    print(f"  {'모델':<35}  {'MAE':>10}  {'RMSE':>10}  {'R²':>8}")
    print(f"  {'─'*67}")
    r2_base     = print_metrics("Ridge 기본 (가중치 없음)", y_va, pred_base)
    r2_weighted = print_metrics("Ridge 가중치 (최근 연도 우선)", y_va, pred_weighted)

    better = "가중치 Ridge" if r2_weighted > r2_base else "기본 Ridge"
    print(f"\n  → R² 기준 더 나은 모델: {better}")
    print(f"     기본 Ridge R²:    {r2_base:.4f}")
    print(f"     가중치 Ridge R²:  {r2_weighted:.4f}  (차이: {r2_weighted - r2_base:+.4f})")

    # ── 백테스팅 저장 ─────────────────────────────────────────────────────
    name_col_bt = '이름' if '이름' in valid_df.columns else valid_df.columns[0]
    bt = valid_df[[name_col_bt, 'AVG_KBO', TARGET]].copy()
    bt = bt.rename(columns={
        name_col_bt: '선수명',
        'AVG_KBO':   '2024_실제타율',
        TARGET:      '2025_실제타율',
    })
    bt['기본Ridge_예측타율']   = np.round(pred_base,     4)
    bt['가중치Ridge_예측타율'] = np.round(pred_weighted, 4)
    bt['기본Ridge_절대오차']   = (bt['기본Ridge_예측타율']   - bt['2025_실제타율']).abs().round(4)
    bt['가중치Ridge_절대오차'] = (bt['가중치Ridge_예측타율'] - bt['2025_실제타율']).abs().round(4)
    bt = bt.sort_values('가중치Ridge_절대오차').reset_index(drop=True)
    bt.index += 1
    bt.to_csv(OUT_BT, index_label='순위', encoding='utf-8-sig')
    print(f"\n백테스팅 결과 저장: {OUT_BT}")

    # ── 전체 데이터로 최종 모델 재학습 ───────────────────────────────────
    X_all, y_all, le_final = prepare_xy(df_labeled, fit_encoder=True)
    w_all = get_sample_weights(df_labeled)

    ridge_final = Ridge(alpha=1.0)
    ridge_final.fit(X_all, y_all, sample_weight=w_all)
    print(f"\n전체 학습 데이터 ({len(df_labeled)}행)로 가중치 Ridge 재학습 완료")

    # ── 2026 타율 예측 ────────────────────────────────────────────────────
    X_pred, _, _ = prepare_xy(df_predict, fit_encoder=False, le=le_final)
    pred_2026 = np.clip(ridge_final.predict(X_pred), 0.0, 1.0)

    # 기본 Ridge 2026 예측 (비교용)
    ridge_base_final = Ridge(alpha=1.0)
    ridge_base_final.fit(X_all, y_all)
    pred_2026_base = np.clip(ridge_base_final.predict(X_pred), 0.0, 1.0)

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    name_col = '이름' if '이름' in df_predict.columns else df_predict.columns[0]

    result = df_predict[[name_col, '연도', 'AVG_KBO', 'PA_KBO', 'OPS', 'WAR',
                          '부상발생여부', '총결장일수', '평균심각도']].copy()
    result = result.rename(columns={
        name_col:    '선수명',
        'AVG_KBO':   '2025_타율(feature)',
        'PA_KBO':    '2025_PA',
        'OPS':       '2025_OPS',
        'WAR':       '2025_WAR',
        '부상발생여부': '2025_부상여부',
        '총결장일수':   '2025_총결장일수',
        '평균심각도':   '2025_평균심각도',
    })
    result['2026_예측타율_기본Ridge']   = np.round(pred_2026_base, 4)
    result['2026_예측타율_가중치Ridge'] = np.round(pred_2026,      4)

    REMOVE_PLAYERS = [
        '케이브', '리베라토', '플로리얼', '카디네스', '위즈덤', '스티븐슨', '맥브룸',
    ]
    result = result[~result['선수명'].isin(REMOVE_PLAYERS)].copy()
    result = result.sort_values('2026_예측타율_가중치Ridge', ascending=False).reset_index(drop=True)
    result.index += 1
    result.to_csv(OUT_PRED, index_label='순위', encoding='utf-8-sig')

    # ── 상위 10명 출력 ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("2026 예측 타율 상위 10명 (가중치 Ridge 기준)")
    print(f"{'='*65}")
    print(f"{'순위':>4}  {'선수명':<8}  {'2025타율':>8}  {'기본Ridge':>10}  {'가중치Ridge':>12}  {'부상':>4}")
    print(f"{'─'*60}")
    for rank, row in result.head(10).iterrows():
        inj = "O" if row['2025_부상여부'] == 1 else "-"
        print(f"{rank:>4}  {str(row['선수명']):<8}  "
              f"{row['2025_타율(feature)']:>8.4f}  "
              f"{row['2026_예측타율_기본Ridge']:>10.4f}  "
              f"{row['2026_예측타율_가중치Ridge']:>12.4f}  {inj:>4}")

    print(f"\n저장 완료: {OUT_PRED}")
    print(f"총 {len(result)}명의 2026 타율 예측값 생성")
    print("=" * 65)


if __name__ == '__main__':
    main()