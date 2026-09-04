"""
KBO 2026 타율 예측 최종 모델
==============================
방법 C (총결장일수 합산 + 평균심각도) 데이터 사용
모델: Ridge Regression (α=1.0)

학습 구조:
  Train    : AVG_KBO_next 있는 전체 행 (2021→2022 ~ 2024→2025, 534행)
  Predict  : 연도 == 2025 & AVG_KBO_next NaN (183행) → 2026 타율 예측

출력:
  kbo_2026_predictions.csv  — 선수별 2026 예측 타율 + feature 정보
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
OUT_PRED   = os.path.join(DATA_DIR, "kbo_2026_predictions.csv")

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


def prepare_xy(df, fit_encoder=True, le=None):
    """feature / target 분리 + 범주형 인코딩"""
    all_feat = STAT_FEATURES + INJURY_FEATURES + CATEGORICAL
    use_feat = [c for c in all_feat if c in df.columns]

    X = df[use_feat].copy()

    # 범주형 인코딩
    if '대표부상분류' in X.columns:
        if fit_encoder:
            le = LabelEncoder()
            X['대표부상분류'] = le.fit_transform(X['대표부상분류'].astype(str))
        else:
            # 학습 시 보지 못한 카테고리 → 0으로 처리
            known = set(le.classes_)
            X['대표부상분류'] = X['대표부상분류'].astype(str).apply(
                lambda v: v if v in known else le.classes_[0]
            )
            X['대표부상분류'] = le.transform(X['대표부상분류'])

    # 수치형 결측 → 0
    X = X.fillna(0)

    y = df[TARGET] if TARGET in df.columns else None
    return X, y, le


def main():
    print("=" * 55)
    print("KBO 2026 타율 예측 — 최종 모델")
    print("=" * 55)

    df = pd.read_csv(INPUT_FILE)
    print(f"\n데이터 로드: {df.shape[0]}행 × {df.shape[1]}열")

    # ── 데이터 분리 ──────────────────────────────────────────────────────────
    df_labeled   = df[df[TARGET].notna()].copy()              # 학습용
    # 연도==2025 인 행만 예측 대상 (은퇴/군입대 등 과거 NaN 행 제외)
    df_predict   = df[(df[TARGET].isna()) & (df['연도'] == 2025)].copy()

    print(f"  학습용 (AVG_KBO_next 있음): {len(df_labeled)}행")
    print(f"  예측용 (2025→2026, 연도==2025): {len(df_predict)}행")

    # ── Validation 성능 확인 (시계열 분할) ──────────────────────────────────
    train_df = df_labeled[df_labeled['연도'] <= 2023]
    valid_df = df_labeled[df_labeled['연도'] == 2024]

    X_tr, y_tr, le = prepare_xy(train_df, fit_encoder=True)
    X_va, y_va, _  = prepare_xy(valid_df, fit_encoder=False, le=le)

    ridge_val = Ridge(alpha=1.0)
    ridge_val.fit(X_tr, y_tr)
    pred_val = ridge_val.predict(X_va)

    mae  = mean_absolute_error(y_va, pred_val)
    rmse = np.sqrt(mean_squared_error(y_va, pred_val))
    r2   = r2_score(y_va, pred_val)

    print(f"\n[검증 성능 — Valid: 2024년 {len(valid_df)}행]")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # ── 백테스팅: 2024년 예측값 vs 실제 2025 타율 비교 ──────────────────
    name_col_bt = '이름' if '이름' in valid_df.columns else valid_df.columns[0]
    bt = valid_df[[name_col_bt, 'AVG_KBO', TARGET]].copy()
    bt = bt.rename(columns={
        name_col_bt: '선수명',
        'AVG_KBO':   '2024_실제타율',
        TARGET:      '2025_실제타율',
    })
    bt['2025_예측타율'] = np.round(pred_val, 4)
    bt['오차'] = np.round(bt['2025_예측타율'] - bt['2025_실제타율'], 4)
    bt['절대오차'] = bt['오차'].abs()
    bt = bt.sort_values('절대오차').reset_index(drop=True)
    bt.index += 1

    OUT_BT = os.path.join(DATA_DIR, "kbo_backtest_2025.csv")
    bt.to_csv(OUT_BT, index_label='순위', encoding='utf-8-sig')

    print(f"\n[백테스팅 — 예측 잘 맞은 상위 10명 (절대오차 작은 순)]")
    print(f"{'순위':>4}  {'선수명':<8}  {'2024실제':>8}  {'2025예측':>8}  {'2025실제':>8}  {'오차':>8}")
    print(f"{'─'*58}")
    for rank, row in bt.head(10).iterrows():
        print(f"{rank:>4}  {str(row['선수명']):<8}  "
              f"{row['2024_실제타율']:>8.4f}  "
              f"{row['2025_예측타율']:>8.4f}  "
              f"{row['2025_실제타율']:>8.4f}  "
              f"{row['오차']:>+8.4f}")
    print(f"\n백테스팅 결과 저장: {OUT_BT}")

    # ── 전체 학습 데이터로 재학습 ─────────────────────────────────────────
    X_all, y_all, le_final = prepare_xy(df_labeled, fit_encoder=True)
    ridge_final = Ridge(alpha=1.0)
    ridge_final.fit(X_all, y_all)
    print(f"\n전체 학습 데이터 ({len(df_labeled)}행)로 최종 모델 재학습 완료")

    # ── 2026 타율 예측 ────────────────────────────────────────────────────
    X_pred, _, _ = prepare_xy(df_predict, fit_encoder=False, le=le_final)
    pred_2026 = ridge_final.predict(X_pred)

    # 타율은 0~1 범위 클리핑
    pred_2026 = np.clip(pred_2026, 0.0, 1.0)

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    # 선수명 컬럼 확인 (이름 또는 선수명)
    name_col = '이름' if '이름' in df_predict.columns else df_predict.columns[0]

    result = df_predict[[name_col, '연도', 'AVG_KBO', 'PA_KBO', 'OPS', 'WAR',
                          '부상발생여부', '총결장일수', '평균심각도']].copy()
    result = result.rename(columns={
        name_col: '선수명',
        'AVG_KBO': '2025_타율(feature)',
        'PA_KBO':  '2025_PA',
        'OPS':     '2025_OPS',
        'WAR':     '2025_WAR',
        '부상발생여부': '2025_부상여부',
        '총결장일수':   '2025_총결장일수',
        '평균심각도':   '2025_평균심각도',
    })
    result['2026_예측타율'] = np.round(pred_2026, 4)

    # ── 외국인 / 은퇴 선수 제거 ──────────────────────────────────────────
    REMOVE_PLAYERS = [
        # 외국인 선수 (2025 시즌 기준)
        '케이브', '리베라토', '플로리얼', '카디네스', '위즈덤', '스티븐슨', '맥브룸',
        # 은퇴 선수 (필요 시 추가)
        # '김현수', '강민호',
    ]
    result = result[~result['선수명'].isin(REMOVE_PLAYERS)].copy()

    # 예측타율 기준 내림차순 정렬
    result = result.sort_values('2026_예측타율', ascending=False).reset_index(drop=True)
    result.index += 1  # 1부터 시작하는 순위

    result.to_csv(OUT_PRED, index_label='순위', encoding='utf-8-sig')

    # ── 상위 10명 출력 ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("2026 예측 타율 상위 10명")
    print(f"{'='*55}")
    print(f"{'순위':>4}  {'선수명':<8}  {'2025타율':>8}  {'2026예측':>8}  {'2025부상':>6}")
    print(f"{'─'*50}")
    for rank, row in result.head(10).iterrows():
        inj = "O" if row['2025_부상여부'] == 1 else "-"
        print(f"{rank:>4}  {str(row['선수명']):<8}  "
              f"{row['2025_타율(feature)']:>8.4f}  "
              f"{row['2026_예측타율']:>8.4f}  {inj:>6}")

    print(f"\n저장 완료: {OUT_PRED}")
    print(f"총 {len(result)}명의 2026 타율 예측값 생성")
    print("=" * 55)

    # ── Feature Importance (Ridge 계수) ──────────────────────────────────
    feat_names = X_all.columns.tolist()
    coefs = pd.Series(ridge_final.coef_, index=feat_names).abs().sort_values(ascending=False)
    print("\n[Ridge 계수 절댓값 — 상위 10개 Feature]")
    for feat, coef in coefs.head(10).items():
        print(f"  {feat:<25} {coef:.4f}")

    # ── 구장별 예측 타율 시뮬레이션 ──────────────────────────────────────
    # KBO 구장별 타율용 파크팩터 (스탯티즈 2025 최종보정 기준)
    # 계산식: 단타×0.63 + 2루타×0.22 + 3루타×0.04 + 홈런×0.11
    # 스케일: 1000 = 중립 (데이터 원본과 동일 스케일)
    STADIUMS = {
        '잠실':   967,  # 두산 / LG
        '문학':   997,  # SSG
        '고척':   963,  # 키움
        '사직':  1006,  # 롯데
        '수원':  1011,  # KT
        '광주':   996,  # KIA
        '대구':  1043,  # 삼성
        '창원':  1022,  # NC
        '대전':  1000,  # 한화
    }

    print(f"\n{'='*55}")
    print("구장별 2026 예측 타율 시뮬레이션")
    print(f"{'='*55}")

    # 예측에 쓸 feature 인덱스 확인
    feat_cols = X_pred.columns.tolist()

    stadium_rows = []
    for idx_pos, (orig_idx, player_row) in enumerate(df_predict.iterrows()):
        player_name = str(player_row[name_col])
        if player_name in REMOVE_PLAYERS:
            continue

        base_feat = X_pred.iloc[idx_pos].copy()
        row_data = {'선수명': player_name, '2025_실제타율': round(player_row['AVG_KBO'], 4)}

        for stadium, pf in STADIUMS.items():
            sim_feat = base_feat.copy()
            if 'Park_Factor' in feat_cols:
                sim_feat['Park_Factor'] = pf
            pred_sim = ridge_final.predict([sim_feat])[0]
            row_data[stadium] = round(float(np.clip(pred_sim, 0.0, 1.0)), 4)

        stadium_rows.append(row_data)

    stadium_df = pd.DataFrame(stadium_rows)
    # 2025 실제 타율 기준 내림차순
    stadium_df = stadium_df.sort_values('2025_실제타율', ascending=False).reset_index(drop=True)
    stadium_df.index += 1

    OUT_STAD = os.path.join(DATA_DIR, "kbo_2026_by_stadium.csv")
    stadium_df.to_csv(OUT_STAD, index_label='순위', encoding='utf-8-sig')

    print(f"상위 10명 미리보기:")
    print(stadium_df.head(10).to_string())
    print(f"\n구장별 예측 저장: {OUT_STAD}")
    print("=" * 55)


if __name__ == '__main__':
    main()