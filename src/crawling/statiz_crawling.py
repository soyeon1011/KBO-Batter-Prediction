import pandas as pd
import sys
import os
from io import StringIO

# ==========================================
# 🎯 연도만 설정하면 10개 구단을 알아서 다 합칩니다!
# target_year = 2021
# target_year = 2022
# target_year = 2023
# target_year = 2024
# target_year = 2025
target_year = 2026
# ==========================================

# KBO 10개 구단 리스트
teams = ['LG', 'KT', 'SSG', 'NC', '두산', 'KIA', '롯데', '삼성', '한화', '키움']

print(f"🔥 [{target_year}년] 10개 구단 HTML 파일 한 번에 합치기 시작...\n")

all_data = [] # 10개 구단 데이터를 모두 담을 큰 바구니

for team_name in teams:
    # 소연님이 저장한 HTML 파일 경로 (예: 2023_LG.html)
    html_file_path = f"/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data/html/{target_year}_{team_name}.html"

    # 파일이 없으면 에러 내지 않고 부드럽게 패스!
    if not os.path.exists(html_file_path):
        print(f"⚠️ 패스: [{team_name}] 팀 파일이 없습니다. ({target_year}_{team_name}.html)")
        continue

    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # pd.read_html, StringIO 사용
        tables = pd.read_html(StringIO(html_content))
        valid_df = None
        
        for tbl in tables:
            if isinstance(tbl.columns, pd.MultiIndex):
                tbl.columns = tbl.columns.droplevel(0)
            if ('이름' in tbl.columns or 'Name' in tbl.columns) and len(tbl) > 0:
                valid_df = tbl
                break

        if valid_df is None or valid_df.empty:
            print(f"🚨 [{team_name}] 표를 찾지 못했습니다. HTML 코드를 확인해주세요.")
            continue

        # 헤더 날리기 ('이름' 또는 'Name')
        name_col = '이름' if '이름' in valid_df.columns else 'Name'
        valid_df = valid_df[valid_df[name_col] != name_col]
        
        valid_df['연도'] = target_year
        valid_df['팀명_스탯티즈'] = team_name

        # 바구니에 담기
        all_data.append(valid_df)
        print(f"✅ [{team_name}] {len(valid_df)}명 데이터 추출 완료!")

    except Exception as e:
        print(f"🚨 [{team_name}] 에러 발생: {e}")

# 반복문이 끝나고 모든 데이터가 모였으면 한 번에 병합해서 저장!
if not all_data:
    print("\n🚨 합칠 데이터가 하나도 없습니다. HTML 파일들을 캡스톤 폴더에 잘 넣으셨는지 확인해주세요!")
    sys.exit()

# 리스트에 모인 10개 구단 데이터프레임을 하나로 합치기
final_df = pd.concat(all_data, ignore_index=True)

# 완벽한 중복 제거
name_col = '이름' if '이름' in final_df.columns else 'Name'
final_df = final_df.drop_duplicates(subset=['연도', name_col, '팀명_스탯티즈'], keep='first')

save_path = f"/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data/statiz_advanced_stats_{target_year}.csv"
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n🎉 대성공!! 10개 구단 총 [{len(final_df)}명]의 데이터를 '{save_path}' 파일 하나로 완벽하게 합쳤습니다!")