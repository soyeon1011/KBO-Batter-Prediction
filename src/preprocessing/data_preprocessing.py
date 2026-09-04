import pandas as pd
import numpy as np

print("📊 23-24-25년 KBO 통합 데이터 전처리 및 부상 지표 생성 시작...")

# 1. 크롤링해 둔 3년 치 데이터 다 불러오기
df_23 = pd.read_csv('/Users/soyeon/Documents/Capstone/data/kbo_hitters_2023_all.csv')
df_24 = pd.read_csv('/Users/soyeon/Documents/Capstone/data/kbo_hitters_2024_all.csv')
df_25 = pd.read_csv('/Users/soyeon/Documents/Capstone/data/kbo_hitters_2025_all.csv')

# 2. 23년, 24년 데이터에서는 비교에 필요한 '타석(PA)' 컬럼만 쏙 빼오기
df_23_subset = df_23[['선수명', '팀명', 'PA']].rename(columns={'PA': 'PA_23'})
df_24_subset = df_24[['선수명', '팀명', 'PA']].rename(columns={'PA': 'PA_24'})

# 3. 최신 데이터인 25년도 엑셀에 24년, 23년 타석 기록을 차례대로 이어 붙이기 (Merge)
merged_df = pd.merge(df_25, df_24_subset, on=['선수명', '팀명'], how='left')
merged_df = pd.merge(merged_df, df_23_subset, on=['선수명', '팀명'], how='left')

# 4. 결측치 처리 (과거에 안 뛰었던 신인 선수들은 타석 수를 0으로 채움)
merged_df['PA_23'] = merged_df['PA_23'].fillna(0)
merged_df['PA_24'] = merged_df['PA_24'].fillna(0)

# 5. 핵심 로직: 연도별 부상(타석 급감) 지표 동시 계산!
# [24년도 부상 마크] 23년 대비 24년 타석 30% 이상 감소
merged_df['타석증감율_24(%)'] = np.where(
    merged_df['PA_23'] >= 100, 
    ((merged_df['PA_24'] - merged_df['PA_23']) / merged_df['PA_23']) * 100, 
    0
)
merged_df['Injury_Flag_24'] = np.where(merged_df['타석증감율_24(%)'] <= -30.0, 1, 0)

# [25년도 부상 마크] 24년 대비 25년(현재 컬럼명 'PA') 타석 30% 이상 감소
merged_df['타석증감율_25(%)'] = np.where(
    merged_df['PA_24'] >= 100, 
    ((merged_df['PA'] - merged_df['PA_24']) / merged_df['PA_24']) * 100, 
    0
)
merged_df['Injury_Flag_25'] = np.where(merged_df['타석증감율_25(%)'] <= -30.0, 1, 0)

# 결과 확인 (최근 25년도에 타석이 급감한 선수들 출력)
print("\n🚨 [AI 분석] 24년 대비 25년 타석 수가 30% 이상 급감한 타자 (부상/에이징커브 의심):")
injury_25 = merged_df[merged_df['Injury_Flag_25'] == 1]
print(injury_25[['선수명', '팀명', 'PA_24', 'PA', '타석증감율_25(%)']].head(10))

# 6. 파생 변수가 모두 추가된 최종 마스터 데이터 저장
save_path = "/Users/soyeon/Documents/Capstone/data/kbo_master_data_3years.csv"
merged_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n🎉 3년 치 통합 전처리 완료! 모든 정보가 담긴 마스터 파일이 '{save_path}'에 저장되었습니다.")