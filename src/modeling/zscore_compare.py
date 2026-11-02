"""
Z-score 변수 추가 효과 비교 (Ridge, alpha=1.0)
- 데이터: kbo_final_dataset_methodC.csv (최종 모델과 같은 파일)
- 전처리: Kbo_final_model.py와 동일 (대표부상분류 레이블 인코딩, 결측 0)
- 평가: 확장 창 순차 평가 (특징 2022→정답 2023, 2023→2024, 2024→2025)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = "/Users/soyeon/Documents/Capstone/data"
df = pd.read_csv(os.path.join(DATA_DIR, "kbo_final_dataset_methodC.csv"))
TARGET = 'AVG_KBO_next'

BASE = ['AVG_KBO', 'PA_KBO', 'AB_KBO', 'H_KBO', 'HR_KBO', 'RBI_KBO',
        'OBP', 'SLG', 'OPS', 'wRC+', 'WAR', 'BABIP', 'ISO', 'wOBA', 'BB/K',
        '데뷔연차', 'Age_2026', 'Park_Factor',
        '부상발생여부', '부상횟수', '총결장일수', '평균심각도', '최고신뢰도점수', '대표부상분류']
Z = ['wOBA_Z', 'OPS_Z', 'WAR_Z', 'BABIP_Z', 'ISO_Z', 'AVG_KBO_Z']
RAW_OF_Z = ['wOBA', 'OPS', 'WAR', 'BABIP', 'ISO', 'AVG_KBO']

FEATURE_SETS = {
    '현재(원래 값만)': BASE,
    '원래 값 + Z-score 6개': BASE + Z,
    '6개를 Z-score로 교체': [c for c in BASE if c not in RAW_OF_Z] + Z,
}


def make_x(d, feats, le=None):
    X = d[feats].copy()
    if le is None:
        le = LabelEncoder().fit(X['대표부상분류'].astype(str))
    known = set(le.classes_)
    X['대표부상분류'] = le.transform(
        X['대표부상분류'].astype(str).map(lambda v: v if v in known else le.classes_[0]))
    return X.fillna(0), le


lab = df[df[TARGET].notna()]
for name, feats in FEATURE_SETS.items():
    maes, r2s = [], []
    for vy in [2022, 2023, 2024]:
        tr, va = lab[lab['연도'] < vy], lab[lab['연도'] == vy]
        Xtr, le = make_x(tr, feats)
        Xva, _ = make_x(va, feats, le)
        pred = Ridge(alpha=1.0).fit(Xtr, tr[TARGET]).predict(Xva)
        maes.append(mean_absolute_error(va[TARGET], pred))
        r2s.append(r2_score(va[TARGET], pred))
    print(f"{name:<18} 2024→2025 MAE={maes[-1]:.4f} R2={r2s[-1]:.4f} | "
          f"연도별 MAE {maes[0]:.4f} {maes[1]:.4f} {maes[2]:.4f}  평균={np.mean(maes):.4f}")