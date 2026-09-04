import pandas as pd

# 1. 뉴스 분석까지 완료된 파일 불러오기
file_path = "/Users/soyeon/Documents/Capstone/data/kbo_injury_final_with_news.csv"
df = pd.read_csv(file_path)

# 🌟 2. 마법의 '0, 1' 변환 (pd.get_dummies)
# 부상분류 컬럼을 쪼개서 '부상_하체_다리', '부상_손_허리' 등의 컬럼으로 만들고 0/1을 채웁니다.
df_final = pd.get_dummies(df, columns=['부상분류'], prefix='부상', dtype=int)

# 3. 결과 확인 (제대로 0, 1이 들어갔는지 상위 5개만 출력)
# 예: 부상_하체_다리, 부상_손_허리 등의 컬럼이 생겼을 거예요.
injury_cols = [col for col in df_final.columns if col.startswith('부상_')]
print("✅ 0과 1로 변환된 부상 컬럼들:")
print(df_final[injury_cols].head())

# 4. 저장 (이제 이 파일이 모델링 직전의 '최종 전처리 파일'이 됩니다)
output_path = "/Users/soyeon/Documents/Capstone/data/kbo_injury_preprocessed.csv"
df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n🎉 변환 완료! 최종 모델링용 부상 파일이 생성되었습니다: {output_path}")