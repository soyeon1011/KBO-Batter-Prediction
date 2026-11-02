"""
KBO 타율 예측 — 방법 B 데이터로 전체 실험 재수행
===================================================
데이터 : kbo_final_dataset_methodB.csv (부상 집계: 결장일수가 가장 긴 1건 + 최대심각도)
분석 단위 : 선수-시즌, t시즌 특징 → t+1시즌 타율(AVG_KBO_next)
전처리 : Kbo_final_model.py와 동일
         - 대표부상분류: 학습 세트 기준 LabelEncoder (검증·예측 세트는 같은 인코더로 변환)
         - 수치형 결측 → 0

출력 (모두 print, 예측값은 CSV로 저장)
  [1] 8개 모델 비교           (특징 2024 → 정답 2025, 표 2)
  [2] 단순 기준선 비교         (+ 선수 단위 부트스트랩 95% CI, 표 3)
  [3] 연도별 순차 평가         (확장 창, 정답 2023/2024/2025, 표 4)
  [4] 특징 제거 실험           (표 5)
  [5] 추가 실험               (A/B/C 집계 비교, Z-score, 앙상블, 연도 가중치)
  [6] 2026 최종 예측           (2021~2024 특징 전체로 재학습 → 2025 특징으로 2026 예측)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 경로 / 설정
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/soyeon/Documents/Capstone/data"
FILE = {m: os.path.join(DATA_DIR, f"kbo_final_dataset_method{m}.csv") for m in "ABC"}
OUT_PRED = os.path.join(DATA_DIR, "kbo_2026_predictions_methodB.csv")
TARGET = 'AVG_KBO_next'
N_BOOT = 5000

BAT = ['AVG_KBO', 'PA_KBO', 'AB_KBO', 'H_KBO', 'HR_KBO', 'RBI_KBO', 'OBP', 'SLG',
       'OPS', 'wRC+', 'WAR', 'BABIP', 'ISO', 'wOBA', 'BB/K']
CAREER = ['데뷔연차', 'Age_2026']
PARK = ['Park_Factor']
INJURY = {
    'A': ['부상발생여부', '부상횟수', '총결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류'],
    'B': ['부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류'],
    'C': ['부상발생여부', '부상횟수', '총결장일수', '평균심각도', '최고신뢰도점수', '대표부상분류'],
}
Z = ['wOBA_Z', 'OPS_Z', 'WAR_Z', 'BABIP_Z', 'ISO_Z', 'AVG_KBO_Z']
ALL_B = BAT + CAREER + PARK + INJURY['B']          # 24개

FOREIGN_2025 = ['케이브', '리베라토', '플로리얼', '카디네스', '위즈덤', '스티븐슨', '맥브룸']


# ─────────────────────────────────────────────────────────────────────────────
# 공통 함수
# ─────────────────────────────────────────────────────────────────────────────
def make_x(d, feats, le=None):
    """학습 세트에서 만든 인코더(le)를 검증·예측 세트에 그대로 사용"""
    X = d[feats].copy()
    if '대표부상분류' in feats:
        if le is None:
            le = LabelEncoder().fit(X['대표부상분류'].astype(str))
        known = set(le.classes_)
        X['대표부상분류'] = le.transform(
            X['대표부상분류'].astype(str).map(lambda v: v if v in known else le.classes_[0]))
    return X.fillna(0), le


def metrics(y, p):
    return (mean_absolute_error(y, p), np.sqrt(mean_squared_error(y, p)), r2_score(y, p))


def split(lab, valid_year):
    """확장 창: valid_year 이전 특징 연도 전부로 학습, valid_year 특징으로 평가"""
    return lab[lab['연도'] < valid_year], lab[lab['연도'] == valid_year]


def fit_predict(model, tr, va, feats, weight=None):
    Xtr, le = make_x(tr, feats)
    Xva, _ = make_x(va, feats, le)
    if weight is None:
        model.fit(Xtr, tr[TARGET])
    else:
        model.fit(Xtr, tr[TARGET], sample_weight=weight)
    return model.predict(Xva)


def league_avg(df):
    g = df.groupby('연도')
    return (g['H_KBO'].sum() / g['AB_KBO'].sum()).to_dict()


def baselines(df, tr, va):
    """예측 시점(특징 연도 말)에 알 수 있는 정보만 사용"""
    yr = int(va['연도'].iloc[0])
    lg = league_avg(df)
    out = {'전년도 타율': va['AVG_KBO'].values,
           '리그 평균': np.full(len(va), lg[yr])}
    hist = df[df['연도'] <= yr].groupby('이름')[['H_KBO', 'AB_KBO']].sum()
    out['과거 통합 타율'] = va['이름'].map(hist['H_KBO'] / hist['AB_KBO']).values
    # 평균회귀: (안타 + k*리그타율) / (타수 + k), k는 학습 구간에서만 선택
    best_k, best_m = 0, np.inf
    lg_tr = tr['연도'].map(lg)
    for k in range(0, 1001, 25):
        m = mean_absolute_error(tr[TARGET], (tr['H_KBO'] + k * lg_tr) / (tr['AB_KBO'] + k))
        if m < best_m:
            best_k, best_m = k, m
    out[f'평균회귀(k={best_k})'] = ((va['H_KBO'] + best_k * lg[yr]) / (va['AB_KBO'] + best_k)).values
    return out


def boot_ci(y, pa, pb, names, seed=0):
    """선수 단위 클러스터 부트스트랩: MAE(a) - MAE(b)의 95% CI"""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    uniq, inv = np.unique(names, return_inverse=True)
    diffs = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        diffs.append(((ea - eb) * w).sum() / w.sum())
    return (ea - eb).mean(), np.percentile(diffs, [2.5, 97.5])


def models_8():
    return {
        'Ridge':            Ridge(alpha=1.0),
        'Lasso':            Lasso(alpha=0.001),
        'ElasticNet':       ElasticNet(alpha=0.001, l1_ratio=0.5),
        'XGBoost':          XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                         random_state=42, verbosity=0),
        'RandomForest':     RandomForestRegressor(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'LightGBM':         LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                          random_state=42, verbose=-1),
        'SVR':              Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1.0))]),
    }


def line(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    data = {m: pd.read_csv(FILE[m]) for m in "ABC"}
    df = data['B']
    lab = df[df[TARGET].notna()].copy()

    print("데이터:", FILE['B'], df.shape)
    for y in range(2021, 2026):
        print(f"  {y}년 특징 행 {int((df['연도'] == y).sum()):>4}  |  정답 있는 행 "
              f"{int(((df['연도'] == y) & df[TARGET].notna()).sum()):>4}")

    tr, va = split(lab, 2024)
    y_va, names_va = va[TARGET].values, va['이름'].values

    # [1] 8개 모델 비교 ──────────────────────────────────────────────────────
    line(f"[1] 모델 비교  (특징 2024 → 정답 2025, 학습 n={len(tr)}, 검증 n={len(va)})")
    preds = {}
    for name, model in models_8().items():
        p = fit_predict(model, tr, va, ALL_B)
        preds[name] = p
        mae, rmse, r2 = metrics(y_va, p)
        print(f"  {name:<18} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    p_lasso_conv = fit_predict(Lasso(alpha=0.001, max_iter=100000), tr, va, ALL_B)
    d, ci = boot_ci(y_va, preds['Ridge'], p_lasso_conv, names_va)
    print(f"  Ridge − Lasso(수렴, MAE={mean_absolute_error(y_va, p_lasso_conv):.4f}) "
          f"MAE 차이 {d:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    d, ci = boot_ci(y_va, preds['Ridge'], preds['ElasticNet'], names_va)
    print(f"  Ridge − ElasticNet MAE 차이 {d:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # [2] 단순 기준선 ───────────────────────────────────────────────────────
    line("[2] 단순 기준선 비교  (특징 2024 → 정답 2025)")
    p_ridge = preds['Ridge']
    mae, rmse, r2 = metrics(y_va, p_ridge)
    print(f"  {'Ridge':<16} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    for name, p in baselines(df, tr, va).items():
        mae, rmse, r2 = metrics(y_va, p)
        d, ci = boot_ci(y_va, p_ridge, p, names_va)
        print(f"  {name:<16} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  "
              f"| Ridge−기준선 {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # [3] 연도별 순차 평가 ──────────────────────────────────────────────────
    line("[3] 연도별 순차 평가  (확장 창)")
    for vy in [2022, 2023, 2024]:
        t, v = split(lab, vy)
        p = fit_predict(Ridge(alpha=1.0), t, v, ALL_B)
        b = baselines(df, t, v)
        rk = [k for k in b if k.startswith('평균회귀')][0]
        mb = {k: mean_absolute_error(v[TARGET], x) for k, x in b.items()}
        print(f"  정답 {vy + 1} (학습 n={len(t)}, 평가 n={len(v)})  "
              f"Ridge={mean_absolute_error(v[TARGET], p):.4f} | {rk}={mb[rk]:.4f} | "
              f"전년도={mb['전년도 타율']:.4f} | 리그평균={mb['리그 평균']:.4f} | "
              f"과거통합={mb['과거 통합 타율']:.4f}")

    # [4] 특징 제거 실험 ────────────────────────────────────────────────────
    line("[4] 특징 제거 실험  (Ridge)")
    sets = {'타격+경력': BAT + CAREER,
            '타격+경력+파크팩터': BAT + CAREER + PARK,
            '타격+경력+부상': BAT + CAREER + INJURY['B'],
            '전체(24개)': ALL_B}
    for name, f in sets.items():
        maes = []
        for vy in [2022, 2023, 2024]:
            t, v = split(lab, vy)
            maes.append(mean_absolute_error(v[TARGET], fit_predict(Ridge(alpha=1.0), t, v, f)))
        mae, rmse, r2 = metrics(y_va, fit_predict(Ridge(alpha=1.0), tr, va, f))
        print(f"  {name:<18} 2025 검증 MAE={mae:.4f} R²={r2:.4f} | 연도별 "
              f"{maes[0]:.4f} {maes[1]:.4f} {maes[2]:.4f}  평균={np.mean(maes):.4f}")

    # [5] 추가 실험 ─────────────────────────────────────────────────────────
    line("[5-1] 부상 집계 방식 A/B/C 비교  (Ridge)")
    for m in "ABC":
        d_m = data[m]
        lab_m = d_m[d_m[TARGET].notna()]
        f = BAT + CAREER + PARK + INJURY[m]
        maes = []
        for vy in [2022, 2023, 2024]:
            t, v = split(lab_m, vy)
            maes.append(mean_absolute_error(v[TARGET], fit_predict(Ridge(alpha=1.0), t, v, f)))
        t, v = split(lab_m, 2024)
        mae, rmse, r2 = metrics(v[TARGET], fit_predict(Ridge(alpha=1.0), t, v, f))
        print(f"  방법 {m}  검증 MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f} | "
              f"연도별 {maes[0]:.4f} {maes[1]:.4f} {maes[2]:.4f}  평균={np.mean(maes):.4f}")

    line("[5-2] 연도별 Z-score 변수  (Ridge)")
    z_sets = {'원래 값만': ALL_B,
              '원래 값 + Z 6개': ALL_B + Z,
              '6개를 Z로 교체': [c for c in ALL_B if c not in
                              ['wOBA', 'OPS', 'WAR', 'BABIP', 'ISO', 'AVG_KBO']] + Z}
    for name, f in z_sets.items():
        maes = []
        for vy in [2022, 2023, 2024]:
            t, v = split(lab, vy)
            maes.append(mean_absolute_error(v[TARGET], fit_predict(Ridge(alpha=1.0), t, v, f)))
        mae, rmse, r2 = metrics(y_va, fit_predict(Ridge(alpha=1.0), tr, va, f))
        print(f"  {name:<16} 2025 검증 MAE={mae:.4f} R²={r2:.4f} | 연도별 "
              f"{maes[0]:.4f} {maes[1]:.4f} {maes[2]:.4f}  평균={np.mean(maes):.4f}")

    line("[5-3] 앙상블 / 연도 가중치  (특징 2024 → 정답 2025)")
    pr, pl, pe = preds['Ridge'], preds['Lasso'], preds['ElasticNet']
    for name, p in [('Ridge 단독', pr), ('Ridge+Lasso+ElasticNet 평균', (pr + pl + pe) / 3),
                    ('Ridge+ElasticNet 평균', (pr + pe) / 2), ('Ridge+Lasso 평균', (pr + pl) / 2)]:
        mae, rmse, r2 = metrics(y_va, p)
        print(f"  {name:<28} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    w = tr['연도'].map({2021: 1, 2022: 2, 2023: 4}).values
    mae, rmse, r2 = metrics(y_va, fit_predict(Ridge(alpha=1.0), tr, va, ALL_B, weight=w))
    print(f"  {'Ridge 연도 가중치(1/2/4)':<28} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # [6] 2026 최종 예측 ────────────────────────────────────────────────────
    line("[6] 2026 최종 예측  (특징 2021~2024 전체로 재학습 → 2025 특징으로 예측)")
    pred_df = df[(df[TARGET].isna()) & (df['연도'] == 2025)].copy()
    X_all, le = make_x(lab, ALL_B)
    final = Ridge(alpha=1.0).fit(X_all, lab[TARGET])
    X_pred, _ = make_x(pred_df, ALL_B, le)
    pred_df['2026_예측타율'] = np.clip(final.predict(X_pred), 0, 1).round(4)

    result = pred_df[['이름', '팀명', 'AVG_KBO', 'PA_KBO', 'AB_KBO', '2026_예측타율']].rename(
        columns={'이름': '선수명', 'AVG_KBO': '2025_타율', 'PA_KBO': '2025_PA', 'AB_KBO': '2025_AB'})
    result = result[~result['선수명'].isin(FOREIGN_2025)]
    result = result.sort_values('2026_예측타율', ascending=False).reset_index(drop=True)
    result.index += 1
    result.to_csv(OUT_PRED, index_label='순위', encoding='utf-8-sig')
    print(f"  학습 {len(lab)}행, 예측 {len(result)}명 (외국인 선수 제외) → {OUT_PRED}")
    print(result.head(10).to_string())


if __name__ == '__main__':
    main()