"""
[4단계] 새 부상 분류 결과로 모델링 데이터(A/B/C) 다시 만들기
==============================================================
입력 : kbo_injury_batter_v32_final.csv       (3단계 결과, 부상 사건 단위)
       kbo_final_dataset_methodB.csv         (기존 모델링 데이터: 타격·경력·파크팩터·정답 열을 그대로 사용)
출력 : kbo_final_dataset_methodA_v2.csv
       kbo_final_dataset_methodB_v2.csv
       kbo_final_dataset_methodC_v2.csv

규칙
  - 실제부상여부 = 0 인 사건(비부상: 부진·조정 말소, 2군 출장 등)은 부상 변수 전체에서 제외
  - 선수-시즌 단위 = (선수, 팀, 부상시작일의 연도)
  - 공통 변수: 부상발생여부(실제 부상 1건 이상), 부상횟수, 최고신뢰도점수, 대표부상분류
      · 최고신뢰도점수: Low 1, Mid·Mid(LLM) 3, High 4 의 시즌 내 최댓값
      · 대표부상분류: 결장일수가 가장 긴 부상의 분류 (부상 없음: 정상)
  - 방법별 변수
      A: 총결장일수(합산)          + 최대심각도
      B: 최대결장일수(가장 긴 1건) + 최대심각도
      C: 총결장일수(합산)          + 평균심각도
  - 부상 기록이 없는 선수-시즌은 모든 부상 변수 0, 대표부상분류 '정상'

선수 식별·나이·데뷔연차·정답(AVG_KBO_next) 다시 만들기
  - 기존: 선수명만으로 다음 시즌 타율을 연결 → 동명이인(김민혁·이원석·이재원)이 서로 섞임
  - 변경: [선수명 + 출생연도]로 연결 (출생연도는 3.5단계에서 다시 확인한 값)
  - 출생연도·입단연도는 3.5단계(네이버·나무위키 재수집) 값으로 교체
  - 나이·데뷔연차는 해당 시즌 기준으로 계산 (새 열 '나이', '데뷔연차')

파크팩터
  - 스탯티즈 구장별 '파크팩터(연도별 보정)'의 단타·2루타·3루타·홈런 값(statiz_pf_yearly_adjusted.csv)을
    해당 시즌 리그 안타 유형 구성비로 가중합하여, 선수의 해당 시즌 소속 구단 홈구장 값을 부여
"""
import os
import numpy as np
import pandas as pd

DATA_DIR    = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
INJURY_FILE = os.path.join(DATA_DIR, "kbo_injury_batter_v32_final.csv")
BASE_FILE   = os.path.join(DATA_DIR, "kbo_final_dataset_methodB.csv")
FIX_FILE    = os.path.join(DATA_DIR, "kbo_player_birth_debut_fixed.csv")   # 3.5단계 결과
PF_FILE     = os.path.join(DATA_DIR, "statiz_pf_yearly_adjusted.csv")      # 스탯티즈 파크팩터(연도별 보정)

# 구단 → 홈구장 (한화는 2025년 신구장 이전)
HOME = {'KIA': '광주', 'KT': '수원', 'LG': '잠실', '두산': '잠실', 'NC': '창원', 'SSG': '문학',
        '롯데': '사직', '삼성': '대구', '키움': '고척'}
# 파크팩터 가중치: 'season' = 해당 시즌 리그 안타 유형 구성비(단타·2루타·3루타·홈런 / 안타)
#                  'fixed'  = 기존 공식 (0.63, 0.22, 0.04, 0.11)
PF_WEIGHTS = 'season'


def add_park_factor(base):
    """스탯티즈 '파크팩터(연도별 보정)'의 단타·2루타·3루타·홈런 값을 가중합 → 해당 시즌 소속 구단 홈구장 값 부여"""
    pf = pd.read_csv(PF_FILE)
    if PF_WEIGHTS == 'season':
        g = base.groupby('연도')[['H_KBO', '2B_KBO', '3B_KBO', 'HR_KBO']].sum()
        w = pd.DataFrame({'w1': (g['H_KBO'] - g['2B_KBO'] - g['3B_KBO'] - g['HR_KBO']) / g['H_KBO'],
                          'w2': g['2B_KBO'] / g['H_KBO'], 'w3': g['3B_KBO'] / g['H_KBO'],
                          'w4': g['HR_KBO'] / g['H_KBO']}).reset_index()
        print("파크팩터 가중치(시즌별 리그 안타 구성비):")
        print(w.round(3).to_string(index=False))
        pf = pf.merge(w, on='연도')
    else:
        pf = pf.assign(w1=0.63, w2=0.22, w3=0.04, w4=0.11)
    pf['Park_Factor'] = (pf['단타'] * pf['w1'] + pf['2루타'] * pf['w2'] +
                         pf['3루타'] * pf['w3'] + pf['홈런'] * pf['w4']).round(1)

    base = base.drop(columns=['Park_Factor'], errors='ignore')
    base['구장'] = base['팀명'].map(HOME)
    base.loc[(base['팀명'] == '한화') & (base['연도'] <= 2024), '구장'] = '대전(구)'
    base.loc[(base['팀명'] == '한화') & (base['연도'] >= 2025), '구장'] = '대전(신)'
    base = base.merge(pf[['구장', '연도', 'Park_Factor']], on=['구장', '연도'], how='left')
    miss = base['Park_Factor'].isna().sum()
    print(f"파크팩터 부여: {len(base) - miss}/{len(base)}행" + (f" (없는 행 {miss}개)" if miss else ""))
    return base.drop(columns=['구장'])

OLD_INJURY_COLS = ['부상발생여부', '부상횟수', '총결장일수', '최대결장일수',
                   '최대심각도', '평균심각도', '최고신뢰도점수', '대표부상분류']
CONF_SCORE = {'Low': 1, 'Mid': 3, 'Mid(LLM)': 3, 'High': 4}

# 출생연도 오류 수정: (이름, 팀명, 연도) → 올바른 출생연도
BIRTH_FIX = {
    ('박해민', 'LG', 2022): 1990, ('박해민', 'LG', 2023): 1990,      # 2026으로 잘못 수집됨
    ('박해민', 'LG', 2024): 1990, ('박해민', 'LG', 2025): 1990,      # (2021 삼성 행은 1990)
    ('이정훈', 'KT', 2025): 1994,                                     # 2000으로 잘못 수집됨 (롯데→KT, 1994년생)
}


def rebuild_target(base):
    """선수 식별·나이·데뷔연차를 바로잡고 다음 시즌 타율(정답)을 다시 연결

    1) BIRTH_FIX로 명백한 출생연도 오류 수정
    2) 3.5단계 결과(kbo_player_birth_debut_fixed.csv)가 있으면 (이름, 기존 출생연도)로 붙여
       출생연도·입단연도를 다시 수집한 값으로 교체 (못 찾은 출생연도는 기존 값 유지, 입단연도는 비움)
    3) 나이 = 연도 − 출생연도 (해당 시즌 기준),  데뷔연차 = 연도 − 입단연도 + 1 (해당 시즌 기준)
    4) 선수 식별자 = 이름_출생연도 → 중복 행 제거 → 다음 시즌 타율 연결
    """
    base = base.copy()
    for (name, team, year), birth in BIRTH_FIX.items():
        m = (base['이름'] == name) & (base['팀명'] == team) & (base['연도'] == year)
        base.loc[m, '출생연도'] = birth

    base['입단연도_새'] = np.nan
    if os.path.exists(FIX_FILE):
        fix = pd.read_csv(FIX_FILE)[['이름', '출생연도_기존', '출생연도_새', '입단연도_새']]
        fix = fix.rename(columns={'출생연도_기존': '출생연도', '입단연도_새': '_입단'})
        base = base.drop(columns=['입단연도_새']).merge(fix, on=['이름', '출생연도'], how='left') \
                   .rename(columns={'_입단': '입단연도_새'})
        changed = base['출생연도_새'].notna() & (base['출생연도_새'] != base['출생연도'])
        print(f"3.5단계 결과 연결: 출생연도가 바뀐 선수 {sorted(base.loc[changed, '이름'].unique().tolist())}")
        base['출생연도'] = base['출생연도_새'].combine_first(base['출생연도'])
        base = base.drop(columns=['출생연도_새'])
    else:
        print(f"⚠ 3.5단계 결과가 없어 기존 출생연도를 쓰고 데뷔연차는 비워 둡니다: {FIX_FILE}")

    base['나이'] = base['연도'] - base['출생연도']
    base['데뷔연차'] = base['연도'] - base['입단연도_새'] + 1
    bad = base['데뷔연차'].notna() & ((base['데뷔연차'] < 1) | (base['데뷔연차'] > 30))
    if bad.any():
        print(f"⚠ 데뷔연차가 1~30 범위를 벗어난 행 {int(bad.sum())}개 → 비움: "
              f"{sorted(base.loc[bad, '이름'].unique().tolist())}")
        base.loc[bad, '데뷔연차'] = np.nan
    print(f"데뷔연차 없는 행: {int(base['데뷔연차'].isna().sum())}개")

    base['선수키'] = base['이름'] + '_' + base['출생연도'].astype('Int64').astype(str)
    n0 = len(base)
    base = base.drop(columns=['AVG_KBO_next']).drop_duplicates(['선수키', '연도', '팀명'])
    print(f"중복 행 제거: {n0} → {len(base)}행")

    nxt = base[['선수키', '연도', 'AVG_KBO', 'AB_KBO']].copy()
    nxt['연도'] -= 1
    nxt['H'] = nxt['AVG_KBO'] * nxt['AB_KBO']           # 한 시즌 두 팀 기록 → 타수 가중 평균
    nxt = nxt.groupby(['선수키', '연도'], as_index=False)[['H', 'AB_KBO']].sum()
    nxt['AVG_KBO_next'] = (nxt['H'] / nxt['AB_KBO']).round(3)
    base = base.merge(nxt[['선수키', '연도', 'AVG_KBO_next']], on=['선수키', '연도'], how='left')
    base.loc[base['연도'] == 2025, 'AVG_KBO_next'] = np.nan
    print(f"정답 있는 행: {int(base['AVG_KBO_next'].notna().sum())}")
    return base


def aggregate(inj):
    rows = []
    for (name, team, year), g in inj.groupby(['선수', '팀', '연도']):
        top = g.sort_values('실제결장일수', ascending=False).iloc[0]
        rows.append({
            '이름': name, '팀명': team, '연도': year,
            '부상발생여부': 1,
            '부상횟수': len(g),
            '총결장일수': g['실제결장일수'].sum(),
            '최대결장일수': top['실제결장일수'],
            '최대심각도': g['심각도점수'].max(),
            '평균심각도': round(g['심각도점수'].mean(), 3),
            '최고신뢰도점수': g['신뢰도'].map(CONF_SCORE).fillna(0).max(),
            '대표부상분류': top['부상분류'],
        })
    return pd.DataFrame(rows)


def main():
    inj = pd.read_csv(INJURY_FILE)
    inj['선수'] = inj['선수'].astype(str).str.split('(').str[0].str.strip()
    inj['연도'] = pd.to_datetime(inj['부상시작일'].astype(str).str.replace('.', '-', regex=False)).dt.year
    inj['실제결장일수'] = pd.to_numeric(inj['실제결장일수'], errors='coerce').fillna(0)

    total = len(inj)
    inj = inj[inj['실제부상여부'] == 1].copy()
    print(f"부상 사건 {total}건 중 실제 부상 {len(inj)}건 사용 (비부상 {total - len(inj)}건 제외)")

    agg = aggregate(inj)
    base = rebuild_target(pd.read_csv(BASE_FILE))
    base = add_park_factor(base)
    base = base.drop(columns=[c for c in OLD_INJURY_COLS], errors='ignore')
    merged = base.merge(agg, on=['이름', '팀명', '연도'], how='left')

    num_cols = [c for c in OLD_INJURY_COLS if c != '대표부상분류']
    merged[num_cols] = merged[num_cols].fillna(0)
    merged['대표부상분류'] = merged['대표부상분류'].fillna('정상')

    matched = merged['부상발생여부'].eq(1).sum()
    print(f"모델링 데이터 {len(merged)}행 중 부상 있는 선수-시즌 {matched}행 "
          f"(부상 집계 {len(agg)}건 중 {matched}건 연결)")

    keep = {
        'A': ['부상발생여부', '부상횟수', '총결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류'],
        'B': ['부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류'],
        'C': ['부상발생여부', '부상횟수', '총결장일수', '평균심각도', '최고신뢰도점수', '대표부상분류'],
    }
    for m, cols in keep.items():
        drop = [c for c in OLD_INJURY_COLS if c not in cols]
        out = merged.drop(columns=drop)
        path = os.path.join(DATA_DIR, f"kbo_final_dataset_method{m}_v2.csv")
        out.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  방법 {m} → {path}  ({out.shape[0]}행 × {out.shape[1]}열)")

    print("\n[방법 B 부상 변수 요약]")
    print(merged[keep['B'][:-1]].describe().round(2).to_string())
    print(merged['대표부상분류'].value_counts().to_string())


if __name__ == '__main__':
    main()
