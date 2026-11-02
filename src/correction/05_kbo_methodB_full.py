"""
KBO 타율 예측 — 방법 B 데이터로 전체 실험 재수행
===================================================
데이터 : kbo_final_dataset_methodB_v2.csv (부상 집계: 결장일수가 가장 긴 1건 + 최대심각도)
         부상 변수는 투수와 같은 방식으로 재분류(injury_v32 1~4단계)한 값
분석 단위 : 선수-시즌, t시즌 특징 → t+1시즌 타율(AVG_KBO_next)
전처리 : Kbo_final_model.py와 동일
         - 대표부상분류: 학습 세트 기준 LabelEncoder (검증·예측 세트는 같은 인코더로 변환)
         - 수치형 결측 → 0

결과는 DATA_DIR/results_methodB/ 폴더에 저장된다.
  [CSV]
  표2_모델비교.csv          8개 모델 (특징 2024 → 정답 2025)
  표3_기준선비교.csv        Ridge vs 단순 기준선 (+ 선수 단위 부트스트랩 95% CI)
  표4_연도별순차평가.csv    확장 창, 정답 2023/2024/2025
  표5_특징제거.csv          특징 구성별 Ridge
  표6_부상집계방식.csv      A/B/C 비교
  표7_Zscore.csv            연도별 Z-score 변수 추가/교체
  표8_앙상블_가중치.csv     앙상블 / 연도 가중치
  검증_2025_선수별.csv      2025 검증 선수별 예측 타율·실제 타율·오차 (Ridge + 기준선)
  예측_2026_선수별.csv      2026 최종 예측 (외국인 선수 제외)
  [PNG]
  그림1_모델비교_MAE.png
  그림2_2025검증_예측vs실제.png
  그림3_연도별순차평가.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic',
                               'Noto Sans CJK KR', 'Noto Serif CJK KR', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────────────────────
# 경로 / 설정
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
VERSION = "_v2"   # 부상 변수 재분류 데이터 (기존 데이터로 돌리려면 "")
OUT_DIR = os.path.join(DATA_DIR, f"results_methodB{VERSION}")
FILE = {m: os.path.join(DATA_DIR, f"kbo_final_dataset_method{m}{VERSION}.csv") for m in "ABC"}
TARGET = 'AVG_KBO_next'
N_BOOT = 5000

BAT = ['AVG_KBO', 'PA_KBO', 'AB_KBO', 'H_KBO', 'HR_KBO', 'RBI_KBO', 'OBP', 'SLG',
       'OPS', 'wRC+', 'WAR', 'BABIP', 'ISO', 'wOBA', 'BB/K']
CAREER = ['데뷔연차', '나이']                  # 해당 시즌 기준 (04단계에서 계산)
PARK = ['Park_Factor']                         # 스탯티즈 연도별 보정 (04단계에서 계산)
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
    return {'MAE': mean_absolute_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)),
            'R2': r2_score(y, p)}


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


def ridge_mae_by_year(lab, feats):
    out = []
    for vy in [2022, 2023, 2024]:
        t, v = split(lab, vy)
        out.append(mean_absolute_error(v[TARGET], fit_predict(Ridge(alpha=1.0), t, v, feats)))
    return out


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
    out['평균회귀'] = ((va['H_KBO'] + best_k * lg[yr]) / (va['AB_KBO'] + best_k)).values
    return out, best_k


def boot_ci(y, pa, pb, names, seed=0):
    """선수 단위 클러스터 부트스트랩: MAE(a) - MAE(b)의 95% CI"""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    uniq, inv = np.unique(names, return_inverse=True)
    diffs = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        diffs.append(((ea - eb) * w).sum() / w.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (ea - eb).mean(), lo, hi


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


def save(table, filename, title):
    """화면 출력 + CSV 저장 (엑셀에서 한글이 깨지지 않도록 utf-8-sig)"""
    table = table.round(4)
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(table.to_string(index=False))
    path = os.path.join(OUT_DIR, filename)
    table.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def active_2026(data_dir):
    """2026 현역 = 2026 등록 명단(4월 수집) ∪ 2026 스탯티즈 기록에 이름이 있는 선수
    (등록 명단에 빠진 선수가 있어 실제 출전 기록으로 보완)"""
    reg = pd.read_csv(os.path.join(data_dir, "kbo_registration_2026.csv"))
    names = set(reg['선수'].astype(str).str.split('(').str[0].str.strip())
    stz = os.path.join(data_dir, "statiz_advanced_stats_2026.csv")
    if os.path.exists(stz):
        names |= set(pd.read_csv(stz)['Name'].astype(str).str.strip())
    return names


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {m: pd.read_csv(FILE[m]) for m in "ABC"}
    df = data['B']
    lab = df[df[TARGET].notna()].copy()

    print("데이터:", FILE['B'], df.shape)
    for y in range(2021, 2026):
        print(f"  {y}년 특징 행 {int((df['연도'] == y).sum()):>4}  |  정답 있는 행 "
              f"{int(((df['연도'] == y) & df[TARGET].notna()).sum()):>4}")

    tr, va = split(lab, 2024)
    y_va, names_va = va[TARGET].values, va['이름'].values

    # [표2] 8개 모델 비교 ───────────────────────────────────────────────────
    preds, rows = {}, []
    for name, model in models_8().items():
        p = fit_predict(model, tr, va, ALL_B)
        preds[name] = p
        rows.append({'모델': name, **metrics(y_va, p)})
    t2 = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
    save(t2, '표2_모델비교.csv',
         f"[표2] 모델 비교  (특징 2024 → 정답 2025, 학습 n={len(tr)}, 검증 n={len(va)})")

    p_lasso_conv = fit_predict(Lasso(alpha=0.001, max_iter=100000), tr, va, ALL_B)
    print("\n  [Ridge와 차순위 모델의 MAE 차이, 선수 단위 부트스트랩 95% CI]")
    for name, p in [('ElasticNet', preds['ElasticNet']), ('Lasso(수렴)', p_lasso_conv)]:
        d, lo, hi = boot_ci(y_va, preds['Ridge'], p, names_va)
        print(f"  Ridge − {name:<12} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    # [표3] 단순 기준선 ─────────────────────────────────────────────────────
    p_ridge = preds['Ridge']
    base, k_best = baselines(df, tr, va)
    rows = [{'예측 방법': 'Ridge', **metrics(y_va, p_ridge),
             'Ridge−기준선 MAE': np.nan, 'CI 하한': np.nan, 'CI 상한': np.nan}]
    for name, p in base.items():
        d, lo, hi = boot_ci(y_va, p_ridge, p, names_va)
        label = f'평균회귀(k={k_best})' if name == '평균회귀' else name
        rows.append({'예측 방법': label, **metrics(y_va, p),
                     'Ridge−기준선 MAE': d, 'CI 하한': lo, 'CI 상한': hi})
    save(pd.DataFrame(rows), '표3_기준선비교.csv', "[표3] 단순 기준선 비교  (특징 2024 → 정답 2025)")

    # 선수별 2025 검증 결과 (kbo_2026_validation_MMDD.csv와 같은 형식 + 기준선)
    v = va[['이름', '팀명', 'AVG_KBO', 'AB_KBO']].copy()
    v.columns = ['선수명', '팀명', '2024_타율', '2024_AB']
    v['2025_예측타율'] = p_ridge.round(4)
    v['2025_실제타율'] = y_va
    v['오차'] = (v['2025_예측타율'] - v['2025_실제타율']).round(4)
    v['절대오차'] = v['오차'].abs()
    v['평균회귀_예측'] = base['평균회귀'].round(4)
    v['평균회귀_절대오차'] = (v['평균회귀_예측'] - v['2025_실제타율']).abs().round(4)
    v = v.sort_values('절대오차').reset_index(drop=True)
    v.index += 1
    path = os.path.join(OUT_DIR, '검증_2025_선수별.csv')
    v.to_csv(path, index_label='순위', encoding='utf-8-sig')
    print(f"\n  선수별 2025 검증 결과 → {path}")

    # [표4] 연도별 순차 평가 ────────────────────────────────────────────────
    rows = []
    for vy in [2022, 2023, 2024]:
        t, vv = split(lab, vy)
        b, k = baselines(df, t, vv)
        row = {'정답 시즌': vy + 1, '학습 n': len(t), '평가 n': len(vv),
               'Ridge': mean_absolute_error(vv[TARGET], fit_predict(Ridge(alpha=1.0), t, vv, ALL_B))}
        for name, p in b.items():
            row[name] = mean_absolute_error(vv[TARGET], p)
        row['평균회귀 k'] = k
        rows.append(row)
    t4 = pd.DataFrame(rows)
    save(t4, '표4_연도별순차평가.csv', "[표4] 연도별 순차 평가  (확장 창, 값 = MAE)")

    # [표5] 특징 제거 실험 ──────────────────────────────────────────────────
    sets = {'타격+경력': BAT + CAREER,
            '타격+경력+파크팩터': BAT + CAREER + PARK,
            '타격+경력+부상': BAT + CAREER + INJURY['B'],
            '전체(24개)': ALL_B}
    rows = []
    for name, f in sets.items():
        m = metrics(y_va, fit_predict(Ridge(alpha=1.0), tr, va, f))
        ys = ridge_mae_by_year(lab, f)
        rows.append({'특징 구성': name, '2025 검증 MAE': m['MAE'], '2025 검증 R2': m['R2'],
                     'MAE 2023': ys[0], 'MAE 2024': ys[1], 'MAE 2025': ys[2],
                     '3개 연도 평균 MAE': np.mean(ys)})
    save(pd.DataFrame(rows), '표5_특징제거.csv', "[표5] 특징 제거 실험  (Ridge)")

    # [표6] 부상 집계 방식 A/B/C ────────────────────────────────────────────
    rows = []
    for m_ in "ABC":
        d_m = data[m_]
        lab_m = d_m[d_m[TARGET].notna()]
        f = BAT + CAREER + PARK + INJURY[m_]
        t, vv = split(lab_m, 2024)
        m = metrics(vv[TARGET], fit_predict(Ridge(alpha=1.0), t, vv, f))
        ys = ridge_mae_by_year(lab_m, f)
        rows.append({'집계 방식': m_, '검증 MAE': m['MAE'], '검증 RMSE': m['RMSE'], '검증 R2': m['R2'],
                     '3개 연도 평균 MAE': np.mean(ys)})
    save(pd.DataFrame(rows), '표6_부상집계방식.csv', "[표6] 부상 집계 방식 A/B/C 비교  (Ridge)")

    # [표7] Z-score ─────────────────────────────────────────────────────────
    z_sets = {'원래 값만': ALL_B,
              '원래 값 + Z 6개': ALL_B + Z,
              '6개를 Z로 교체': [c for c in ALL_B if c not in
                              ['wOBA', 'OPS', 'WAR', 'BABIP', 'ISO', 'AVG_KBO']] + Z}
    rows = []
    for name, f in z_sets.items():
        m = metrics(y_va, fit_predict(Ridge(alpha=1.0), tr, va, f))
        ys = ridge_mae_by_year(lab, f)
        rows.append({'입력 구성': name, '2025 검증 MAE': m['MAE'], '2025 검증 R2': m['R2'],
                     '3개 연도 평균 MAE': np.mean(ys)})
    save(pd.DataFrame(rows), '표7_Zscore.csv', "[표7] 연도별 Z-score 변수  (Ridge)")

    # [표8] 앙상블 / 연도 가중치 ────────────────────────────────────────────
    pr, pl, pe = preds['Ridge'], preds['Lasso'], preds['ElasticNet']
    w = tr['연도'].map({2021: 1, 2022: 2, 2023: 4}).values
    rows = [{'방법': n, **metrics(y_va, p)} for n, p in [
        ('Ridge 단독', pr),
        ('Ridge+Lasso+ElasticNet 평균', (pr + pl + pe) / 3),
        ('Ridge+ElasticNet 평균', (pr + pe) / 2),
        ('Ridge+Lasso 평균', (pr + pl) / 2),
        ('Ridge 연도 가중치(1/2/4)', fit_predict(Ridge(alpha=1.0), tr, va, ALL_B, weight=w))]]
    save(pd.DataFrame(rows), '표8_앙상블_가중치.csv', "[표8] 앙상블 / 연도 가중치  (특징 2024 → 정답 2025)")

    # 2026 최종 예측 ────────────────────────────────────────────────────────
    pred_df = df[(df[TARGET].isna()) & (df['연도'] == 2025)].copy()
    X_all, le = make_x(lab, ALL_B)
    final = Ridge(alpha=1.0).fit(X_all, lab[TARGET])
    X_pred, _ = make_x(pred_df, ALL_B, le)
    pred_df['2026_예측타율'] = np.clip(final.predict(X_pred), 0, 1).round(4)
    result = pred_df[['이름', '팀명', 'AVG_KBO', 'PA_KBO', 'AB_KBO', '2026_예측타율']].rename(
        columns={'이름': '선수명', 'AVG_KBO': '2025_타율', 'PA_KBO': '2025_PA', 'AB_KBO': '2025_AB'})
    active = active_2026(DATA_DIR)
    result = result[~result['선수명'].isin(FOREIGN_2025) & result['선수명'].isin(active)]
    result = result.sort_values('2026_예측타율', ascending=False).reset_index(drop=True)
    result.index += 1
    path = os.path.join(OUT_DIR, '예측_2026_선수별.csv')
    result.to_csv(path, index_label='순위', encoding='utf-8-sig')
    print(f"\n[2026 최종 예측] 학습 {len(lab)}행, 예측 {len(result)}명 (외국인·2026 미등록 선수 제외) → {path}")
    print(result.head(10).to_string())

    # ── 그림 ──────────────────────────────────────────────────────────────
    # 그림1: 모델별 MAE (낮을수록 좋음)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    order = t2.sort_values('MAE', ascending=False)
    colors = ['#4a6fa5' if n == 'Ridge' else '#b8bec8' for n in order['모델']]
    ax.barh(order['모델'], order['MAE'], color=colors)
    for y_, val in enumerate(order['MAE']):
        ax.text(val + 0.0002, y_, f'{val:.4f}', va='center', fontsize=8)
    ax.set_xlim(0.025, order['MAE'].max() + 0.002)
    ax.set_xlabel('MAE (낮을수록 좋음)')
    ax.set_title('모델별 검증 MAE (특징 2024 → 정답 2025, n=126)', fontsize=10)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '그림1_모델비교_MAE.png'))
    plt.close()

    # 그림2: 2025 검증 예측 vs 실제
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.scatter(v['2025_예측타율'], v['2025_실제타율'], s=14, color='#4a6fa5', alpha=.75, edgecolor='none')
    lo = np.floor(min(v['2025_예측타율'].min(), v['2025_실제타율'].min()) * 50) / 50
    hi = np.ceil(max(v['2025_예측타율'].max(), v['2025_실제타율'].max()) * 50) / 50
    ax.plot([lo, hi], [lo, hi], color='#888', lw=1, ls='--', label='예측 = 실제')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('예측 타율 (Ridge)')
    ax.set_ylabel('2025 실제 타율')
    ax.set_title(f"2025 검증: 예측 vs 실제 (n={len(v)}, MAE={v['절대오차'].mean():.4f})", fontsize=10)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '그림2_2025검증_예측vs실제.png'))
    plt.close()

    # 그림3: 연도별 순차 평가
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
    styles = {'Ridge': ('#4a6fa5', 2.2), '평균회귀': ('#888888', 1.2), '과거 통합 타율': ('#aaaaaa', 1.2),
              '전년도 타율': ('#c4c4c4', 1.2), '리그 평균': ('#d8d8d8', 1.2)}
    for name, (c, lw) in styles.items():
        ax.plot(t4['정답 시즌'], t4[name], marker='o', color=c, lw=lw, label=name)
    ax.set_xticks(t4['정답 시즌'])
    ax.set_ylabel('MAE (낮을수록 좋음)')
    ax.set_xlabel('정답 시즌')
    ax.set_title('연도별 순차 평가: Ridge vs 단순 기준선', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=.25)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '그림3_연도별순차평가.png'))
    plt.close()

    print(f"\n그림 3개 저장 → {OUT_DIR}")


if __name__ == '__main__':
    main()
