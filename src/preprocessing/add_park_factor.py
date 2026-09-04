import pandas as pd

print("🏟️ [구장 효과(Park Factor) 매핑] 데이터 컬럼명 불일치 수정 버전을 실행합니다!\n")

# 파일 경로
file_path = '/Users/soyeon/Documents/Capstone/data/KBO_Master_Data_2021_2025.csv'
df = pd.read_csv(file_path)

# KBO 홈구장 파크 팩터 (득점 기준, 1000이 리그 평균)
def get_park_factor(team, year):
    # 1. 잠실: 투수 친화 (두산, LG)
    if team in ['두산', 'LG']:
        return 930
    
    # 2. 대구/인천: 타자 친화 (삼성, SSG)
    elif team == '삼성':
        return 1060
    elif team == 'SSG':
        return 1050
        
    # 3. 창원/수원/광주 (NC, KT, KIA)
    elif team == 'NC':
        return 1035
    elif team == 'KT':
        return 1025
    elif team == 'KIA':
        return 1015
        
    # 4. 고척돔: 투수 친화 (키움)
    elif team == '키움':
        return 975
        
    # 5. 사직: 2022년 펜스 높임 (롯데)
    elif team == '롯데':
        # 🚨 여기서도 x['연도']가 넘어오므로 year 변수를 사용합니다.
        if year <= 2021:
            return 1010
        else:
            return 965
            
    # 6. 대전: 2025년 신축 (한화)
    elif team == '한화':
        if year >= 2025:
            return 1000
        else:
            return 990
            
    return 1000

# 🚨 [수정 포인트] x['시즌']을 x['연도']로 변경했습니다.
df['Park_Factor'] = df.apply(lambda x: get_park_factor(x['팀명'], x['연도']), axis=1)

# 엑셀 저장
df.to_csv(file_path, index=False, encoding='utf-8-sig')

print("🎉 [매핑 성공] 전원의 구장 효과(Park Factor)가 엑셀에 추가되었습니다!")

# 🚨 [수정 포인트] 출력문에서도 '시즌' 대신 '연도'를 사용합니다.
print(df[['이름', '팀명', '연도', 'Park_Factor']].head(10).to_string(index=False))