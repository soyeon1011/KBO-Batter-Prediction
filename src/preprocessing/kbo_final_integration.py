import pandas as pd

# 1. 파일 경로 (소연님 맥북 실제 경로로 확인)
injury_path = "/Users/soyeon/Documents/Capstone/data/kbo_injury_final_with_news.csv"
master_path = "/Users/soyeon/Documents/Capstone/data/KBO_Master_Data_2021_2025_final.csv"

# 2. 데이터 로드
injury_df = pd.read_csv(injury_path)
master_df = pd.read_csv(master_path)

# 🌟 3. 부상 데이터 전처리 (합치기 좋게 다듬기)
injury_df['연도'] = pd.to_datetime(injury_df['부상시작일']).dt.year
injury_df['선수_정제'] = injury_df['선수'].str.split('(').str[0].str.strip()

# 🌟 4. [핵심] 마스터 데이터를 '왼쪽(Base)'으로 두고 병합!
# 이렇게 해야 875명의 성적 데이터가 하나도 안 사라집니다.
final_df = pd.merge(
    master_df, 
    injury_df, 
    left_on=['연도', '이름'], 
    right_on=['연도', '선수_정제'], 
    how='left'
)

# 🌟 5. 안 다친 선수들(NaN) 처리
# 부상 관련 수치형 데이터는 0으로, 범주형은 '정상'으로 채워줍니다.
final_df['실제결장일수'] = final_df['실제결장일수'].fillna(0)
final_df['심각도점수'] = final_df['심각도점수'].fillna(0)
final_df['재검필요'] = final_df['재검필요'].fillna(0)

# 부상 분류 더미(0/1) 컬럼들도 안 다친 선수는 0으로 채우기
injury_cols = [col for col in final_df.columns if col.startswith('부상_')]
final_df[injury_cols] = final_df[injury_cols].fillna(0)

# 6. 최종 통합 데이터 저장
save_path = "/Users/soyeon/Documents/Capstone/data/kbo_integrated_final_data.csv"
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"🎉 [통합 완료] 총 {len(final_df)}건의 데이터가 생성되었습니다. (성적 데이터 100% 보존)")
print(f"✅ 최형우 선수 데이터 확인: {len(final_df[final_df['이름'] == '최형우'])}건 발견")
print(f"💾 저장 위치: {save_path}")