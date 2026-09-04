import pandas as pd

# 1. 통합 데이터 로드
file_path = "/Users/soyeon/Documents/Capstone/data/kbo_integrated_final_data.csv"
df = pd.read_csv(file_path)

# 🌟 2. 한글 '부상분류'를 0과 1의 숫자 컬럼으로 변환 (One-Hot Encoding)
# '부상_하체_다리', '부상_손_허리' 처럼 새로운 칸들이 생깁니다.
df_dummies = pd.get_dummies(df, columns=['부상분류'], prefix='부상')

# 🌟 3. 불리언(True/False) 대신 숫자(1/0)로 확실히 고정
# XGBoost 같은 모델은 True/False보다 1/0을 더 좋아합니다.
injury_cols = [col for col in df_dummies.columns if col.startswith('부상_')]
df_dummies[injury_cols] = df_dummies[injury_cols].astype(int)

# 4. 안 다친 선수들(정상) 처리
# 부상 관련 정보가 없는 성적 데이터들은 결장일수와 심각도를 0으로 채워줍니다.
df_dummies['실제결장일수'] = df_dummies['실제결장일수'].fillna(0)
df_dummies['심각도점수'] = df_dummies['심각도점수'].fillna(0)

# 5. 최종 모델링용 파일로 저장
save_path = "/Users/soyeon/Documents/Capstone/data/kbo_modeling_final.csv"
df_dummies.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"✅ 한글 부상분류를 0/1 숫자로 변환 완료!")
print(f"📊 생성된 컬럼: {injury_cols}")
print(f"💾 최종 학습용 파일 저장: {save_path}")