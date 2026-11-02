"""
[9단계] 2026 독립 시험 — 표본 기준 민감도 + 사전에 정한 집단별 오차
=====================================================================
교수님 검토 의견 8번 반영
  - PA와 AB를 함께 보고하고, 표본 기준을 바꿔도 결과가 유지되는지 확인
  - 사전에 정한 연령·부상 여부·이적 여부별 오차

입력 : results_methodB_v2/예측_2026_선수별.csv   (05단계)
       statiz_advanced_stats_2026.csv             (시즌 종료 후 갱신본)
       kbo_final_dataset_methodB_v2.csv           (2025시즌 나이·부상 여부)
       results_methodB_v2/표4_연도별순차평가.csv  (평균회귀 k)
출력 : results_methodB_v2/표11_2026_표본기준별.csv
       results_methodB_v2/표12_2026_집단별.csv
"""
import os
import numpy as np
import pandas as pd

DATA_DIR  = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
OUT_DIR   = os.path.join(DATA_DIR, "results_methodB_v2")
PRED_FILE = os.path.join(OUT_DIR, "예측_2026_선수별.csv")
STATIZ    = os.path.join(DATA_DIR, "statiz_advanced_stats_2026.csv")
B_FILE    = os.path.join(DATA_DIR, "kbo_final_dataset_methodB_v2.csv")
T4_FILE   = os.path.join(OUT_DIR, "표4_연도별순차평가.csv")
TRADED_2026 = ['하주석', '데이비슨']   # 2026 시즌 중 이적 (두 팀 기록 합산)
MIN_PA = 50                           # 본문 기준

# 집단 경계는 결과를 보기 전에 정한 값 (나이 구간은 초안 Table 6과 동일)
AGE_BINS   = [0, 27, 31, 100];   AGE_LABELS = ['27세 이하', '28~31세', '32세 이상']
# 출전량 구간은 예측 시점에 알 수 있는 2025시즌 타석으로 나눈다
# (2026 타석으로 나누면 2026에 잘 친 선수가 출전이 늘어 많은 타석 구간에 몰리므로 사후 선별이 됨)
PA25_BINS  = [0, 200, 400, 1000]; PA25_LABELS = ['200 이하', '201~400', '401 이상']


def load():
    pred = pd.read_csv(PRED_FILE)                       # 선수명, 팀명(2025), 2025_타율, 2025_AB, 2026_예측타율 ...
    stz = pd.read_csv(STATIZ)
    stz['Name'] = stz['Name'].astype(str).str.strip()

    # 2026 소속 팀 목록 (이적 여부 판단용)
    teams26 = stz.groupby('Name')['팀명_스탯티즈'].apply(lambda s: sorted(set(s))).to_dict()

    # 시즌 중 이적 선수는 기록 합산
    trd = stz[stz['Name'].isin(TRADED_2026)].groupby('Name', as_index=False)[['PA', 'AB', 'H']].sum()
    trd['AVG'] = trd['H'] / trd['AB']
    stz = pd.concat([stz[~stz['Name'].isin(TRADED_2026)][['Name', 'PA', 'AB', 'H', 'AVG']], trd],
                    ignore_index=True)
    dup = stz['Name'][stz['Name'].duplicated(keep=False)].unique()
    if len(dup):
        print(f"동명이인으로 제외: {sorted(dup)}")
    stz = stz[~stz['Name'].isin(dup)].rename(columns={
        'Name': '선수명', 'PA': '2026_PA', 'AB': '2026_AB', 'H': '2026_H', 'AVG': '2026_실제타율'})

    df = pred.merge(stz, on='선수명', how='inner')

    # 2025시즌 나이·부상 여부 (이름 + 2025 소속팀으로 연결)
    b = pd.read_csv(B_FILE)
    b25 = b[b['연도'] == 2025][['이름', '팀명', '나이', '부상발생여부', 'H_KBO', 'AB_KBO']]
    df = df.merge(b25.rename(columns={'이름': '선수명'}), on=['선수명', '팀명'], how='left')
    df['2026_나이'] = df['나이'] + 1
    df['2025_부상'] = np.where(df['부상발생여부'] == 1, '있음', '없음')

    # 이적: 2026 소속팀이 2025 소속팀과 다르거나, 2026 시즌 중 팀을 옮긴 경우
    def moved(r):
        t = teams26.get(r['선수명'], [])
        return '있음' if (len(t) > 1 or (len(t) == 1 and t[0] != r['팀명'])) else '없음'
    df['이적'] = df.apply(moved, axis=1)

    # 기준선 (예측 시점에 알 수 있는 2025 기록)
    lg25 = b[b['연도'] == 2025]['H_KBO'].sum() / b[b['연도'] == 2025]['AB_KBO'].sum()
    k = int(pd.read_csv(T4_FILE).iloc[-1]['평균회귀 k']) if os.path.exists(T4_FILE) else 150
    df['평균회귀'] = (df['H_KBO'] + k * lg25) / (df['AB_KBO'] + k)
    df['리그 평균'] = lg25
    df['전년도 타율'] = df['2025_타율']

    y = df['2026_실제타율']
    df['e_Ridge'] = (df['2026_예측타율'] - y).abs()
    for c in ['평균회귀', '리그 평균', '전년도 타율']:
        df[f'e_{c}'] = (df[c] - y).abs()
    return df, k


def main():
    df, k = load()
    cols = ['Ridge', '평균회귀', '리그 평균', '전년도 타율']

    # ── 표11: 표본 기준을 바꿔도 결과가 유지되는가 (PA 기준·AB 기준) ──
    rows = []
    for kind, col, cuts in [('PA', '2026_PA', [30, 50, 100, 200]), ('AB', '2026_AB', [50, 100, 200])]:
        for c in cuts:
            d = df[df[col] >= c]
            r = {'표본 기준': f'{kind} ≥ {c}', '선수 수': len(d),
                 '평균 PA': round(d['2026_PA'].mean()), '평균 AB': round(d['2026_AB'].mean())}
            r.update({f'{m} MAE': round(d[f'e_{m}'].mean(), 4) for m in cols})
            r['Ridge 순위'] = int(pd.Series({m: d[f'e_{m}'].mean() for m in cols}).rank()['Ridge'])
            rows.append(r)
    t11 = pd.DataFrame(rows)
    t11.columns = [c.replace('평균회귀 MAE', f'평균회귀(k={k}) MAE') for c in t11.columns]
    print("[표11] 표본 기준별 2026 MAE"); print(t11.to_string(index=False))
    t11.to_csv(os.path.join(OUT_DIR, '표11_2026_표본기준별.csv'), index=False, encoding='utf-8-sig')

    # ── 표12: 사전에 정한 집단별 오차 (본문 기준 PA ≥ 50) ──
    d = df[df['2026_PA'] >= MIN_PA].copy()
    miss = d['나이'].isna().sum()
    if miss:
        print(f"\n⚠ 2025 나이·부상 정보를 연결하지 못한 선수 {miss}명 (집단별 표에서 '정보 없음')")
    d['2025 타석(PA)'] = pd.cut(d['2025_PA'], PA25_BINS, labels=PA25_LABELS)
    d['2026 나이'] = pd.cut(d['2026_나이'], AGE_BINS, labels=AGE_LABELS)
    d['2025 부상 여부'] = d['2025_부상']
    d['이적 여부'] = d['이적']

    rows = []
    for g in ['2025 타석(PA)', '2026 나이', '2025 부상 여부', '이적 여부']:
        for lv, sub in d.groupby(g, observed=True):
            rows.append({'구분': g, '집단': lv, '선수 수': len(sub),
                         'Ridge MAE': round(sub['e_Ridge'].mean(), 4),
                         '평균회귀 MAE': round(sub['e_평균회귀'].mean(), 4),
                         '평균 편향(예측−실제)': round((sub['2026_예측타율'] - sub['2026_실제타율']).mean(), 4)})
    t12 = pd.DataFrame(rows)
    print(f"\n[표12] 집단별 2026 오차 (PA ≥ {MIN_PA}, n={len(d)})"); print(t12.to_string(index=False))
    t12.to_csv(os.path.join(OUT_DIR, '표12_2026_집단별.csv'), index=False, encoding='utf-8-sig')
    print(f"\n이적 선수: {sorted(d.loc[d['이적'] == '있음', '선수명'])}")
    print(f"저장 → {OUT_DIR}")


if __name__ == '__main__':
    main()