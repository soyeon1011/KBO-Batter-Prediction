import pandas as pd
import os

# 1. 경로 설정
BASE_DIR = "/Users/soyeon/Documents/Capstone/data"
master_path = os.path.join(BASE_DIR, "KBO_Master_Data_2021_2025_final.csv")
injury_path = os.path.join(BASE_DIR, "kbo_integrated_final_data.csv")

# 2. 데이터 불러오기
if not os.path.exists(master_path) or not os.path.exists(injury_path):
    print("❌ 에러: CSV 파일 경로를 확인해 주세요!")
else:
    df_master = pd.read_csv(master_path)
    df_injury = pd.read_csv(injury_path)

    # 3. 데이터 병합 (이름+팀명+연도 기준)
    # 성적 데이터(Master) 전체에 부상 정보(Injury)를 덧붙입니다.
    df_combined = pd.merge(df_master, 
                           df_injury[['선수', '팀', '연도', '부상분류', '실제결장일수', '심각도점수', '재검필요']], 
                           left_on=['이름', '팀명', '연도'], 
                           right_on=['선수', '팀', '연도'], 
                           how='left')

    # 4. 부상 없는 선수들 빈칸 채우기
    df_combined['실제결장일수'] = df_combined['실제결장일수'].fillna(0)
    df_combined['심각도점수'] = df_combined['심각도점수'].fillna(0)
    df_combined['재검필요'] = df_combined['재검필요'].fillna(0)
    df_combined['부상분류'] = df_combined['부상분류'].fillna('정상')

    # 5. 수치화 (One-Hot Encoding)
    # 부상분류를 0과 1로 바꾸되, 원본 한글 데이터와 '이름'은 그대로 유지합니다.
    df_modeling = pd.get_dummies(df_combined, columns=['부상분류'], prefix='부상', dtype=int)

    # 6. 중복된 식별 컬럼만 정리 (이름, 팀명은 남겨둠)
    if '선수' in df_modeling.columns: df_modeling = df_modeling.drop(columns=['선수'])
    if '팀' in df_modeling.columns: df_modeling = df_modeling.drop(columns=['팀'])

    # 7. 최종 통합 파일 저장 (이름 포함 버전)
    output_path = os.path.join(BASE_DIR, "kbo_final_modeling_full.csv")
    df_modeling.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✨ [데이터 보존 완료] 이름과 모든 성적이 포함된 파일이 생성되었습니다.")
    print(f"💾 저장 위치: {output_path}")
    print(f"📊 총 {df_modeling.shape[1]}개의 컬럼이 안전하게 보관되었습니다.")