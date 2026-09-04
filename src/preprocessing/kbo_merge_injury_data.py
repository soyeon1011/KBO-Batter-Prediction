import pandas as pd
import glob
import numpy as np

print("⏳ 1. 데이터 불러오는 중...")

# 1. 5년 치 부상자 데이터 불러오기
injury_files = glob.glob("/Users/soyeon/Documents/Capstone/data/kbo_pure_injury_*.csv")
injury_df = pd.concat([pd.read_csv(f) for f in injury_files], ignore_index=True)

# 부상자 명단의 이름에 있는 괄호 떼기 (예: '김도영(내야수)' -> '김도영')
injury_df['선수'] = injury_df['선수'].apply(lambda x: str(x).split('(')[0].strip())
injury_df['부상시작일'] = pd.to_datetime(injury_df['날짜'])

# 2. 5년 치 1군 엔트리(복귀) 데이터 불러오기
entry_files = glob.glob("/Users/soyeon/Documents/Capstone/data/kbo_entry_history_*.csv")
entry_df = pd.concat([pd.read_csv(f) for f in entry_files], ignore_index=True)

# 쓰레기 데이터('당일 1군 등록 된 선수가 없습니다') 날리기
entry_df = entry_df[~entry_df['팀'].str.contains('없습니다', na=False)]

# '등록' 기록만 따로 빼기 (이게 우리의 '복귀일' 후보들입니다)
callup_df = entry_df[entry_df['상태'] == '등록'].copy()
callup_df['복귀일'] = pd.to_datetime(callup_df['일자'])

print("⏳ 2. 선수별 부상 시작일과 복귀일 짝맞추는 중... (약간의 시간이 걸릴 수 있습니다)")

# 3. 마법의 짝맞추기 로직
def find_return_date(row):
    player = row['선수']
    team = row['팀']
    start_date = row['부상시작일']
    
    # 조건: 같은 팀, 같은 선수이면서, 복귀일이 부상시작일 '이후'인 1군 등록 기록 전부 찾기
    future_callups = callup_df[
        (callup_df['선수'] == player) & 
        (callup_df['팀'] == team) & 
        (callup_df['복귀일'] > start_date)
    ]
    
    if not future_callups.empty:
        # 그중 가장 '먼저' 1군에 올라온 날짜가 진짜 복귀일!
        return future_callups.sort_values('복귀일').iloc[0]['복귀일']
    else:
        # 시즌 아웃으로 그 해에 못 돌아왔거나 방출된 경우
        return pd.NaT

# 판다스 apply로 전체 부상 기록에 함수 적용
injury_df['실제복귀일'] = injury_df.apply(find_return_date, axis=1)

# 4. 실제 결장 일수 계산! (복귀일 - 부상시작일)
injury_df['실제결장일수'] = (injury_df['실제복귀일'] - injury_df['부상시작일']).dt.days

# 보기 좋게 컬럼 순서 정리
final_df = injury_df[['팀', '선수', '항목', '부상시작일', '실제복귀일', '실제결장일수', '부상_일수_텍스트']]
final_df = final_df.rename(columns={'부상_일수_텍스트': 'KBO공식_부상명단일수'})

# 날짜순으로 정렬
final_df = final_df.sort_values(by=['부상시작일']).reset_index(drop=True)

# 5. 최종 마스터 데이터 저장
save_path = "/Users/soyeon/Documents/Capstone/data/kbo_injury_master_dataset.csv"
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"🎉 완벽 성공! 모든 부상 기록의 실제 결장 일수가 계산되었습니다.")
print(f"📁 마스터 데이터 저장 완료: {save_path}")