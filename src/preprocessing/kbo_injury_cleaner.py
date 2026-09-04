import pandas as pd

# 1. 아까 긁어둔 원본 데이터 불러오기
# file_path = "/Users/soyeon/Documents/Capstone/data/kbo_roster_moves_2021.csv"
# file_path = "/Users/soyeon/Documents/Capstone/data/kbo_roster_moves_2022.csv"
# file_path = "/Users/soyeon/Documents/Capstone/data/kbo_roster_moves_2023.csv"
# file_path = "/Users/soyeon/Documents/Capstone/data/kbo_roster_moves_2024.csv"
file_path = "/Users/soyeon/Documents/Capstone/data/kbo_roster_moves_2025.csv"
df = pd.read_csv(file_path)

# 2. '항목' 컬럼에서 진짜 부상/재활 관련 키워드만 남기고 싹 필터링
# '소속선수 추가 등록' 같은 노이즈는 여기서 완벽하게 걸러집니다.
pure_injury_df = df[df['항목'].str.contains('부상|치료|재활', na=False)].copy()

# 3. 데이터 깔끔하게 정렬 (팀명 -> 선수명 -> 날짜 순)
pure_injury_df = pure_injury_df.sort_values(by=['팀', '선수', '날짜']).reset_index(drop=True)

# 4. 정제된 파일로 새로 저장
# save_path = "/Users/soyeon/Documents/Capstone/data/kbo_pure_injury_2021.csv"
# save_path = "/Users/soyeon/Documents/Capstone/data/kbo_pure_injury_2022.csv"
# save_path = "/Users/soyeon/Documents/Capstone/data/kbo_pure_injury_2023.csv"
# save_path = "/Users/soyeon/Documents/Capstone/data/kbo_pure_injury_2024.csv"
save_path = "/Users/soyeon/Documents/Capstone/data/kbo_pure_injury_2025.csv"
pure_injury_df.to_csv(save_path, index=False, encoding='utf-8-sig')

# 결과 확인
print(f"🗑️ 노이즈 데이터 제거 완료!")
print(f"✅ 순수 부상 기록 [ {len(pure_injury_df)}건 ] 추출 성공!")
print(f"📁 저장 완료: {save_path}")