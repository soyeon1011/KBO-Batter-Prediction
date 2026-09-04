import pandas as pd
import numpy as np

# 1. 21~25년도 반복해서 합치기 위한 준비
years = [2021, 2022, 2023, 2024, 2025]
all_years_data = []

print("🔥 5개년 마스터 데이터 병합 및 세이버메트릭스 계산 시작...\n")

for year in years:
    try:
        # 파일 불러오기
        kbo_df = pd.read_csv(f'/Users/soyeon/Documents/Capstone/data/kbo_hitters_{year}_all.csv')
        statiz_df = pd.read_csv(f'/Users/soyeon/Documents/Capstone/data/statiz_advanced_stats_{year}.csv')

        # 컬럼명 통일 (선수명 -> 이름, 팀명_스탯티즈 -> 팀명)
        if '선수명' in kbo_df.columns:
            kbo_df.rename(columns={'선수명': '이름'}, inplace=True)
        if 'Name' in statiz_df.columns:
            statiz_df.rename(columns={'Name': '이름'}, inplace=True)
        if '팀명_스탯티즈' in statiz_df.columns:
            statiz_df.rename(columns={'팀명_스탯티즈': '팀명'}, inplace=True)

        # 🚨 [핵심 수정] '이름'과 '팀명' 두 가지를 모두 만족할 때만 이어 붙이기 (동명이인 크로스 방지!)
        merged_df = pd.merge(kbo_df, statiz_df, on=['이름', '팀명'], how='inner', suffixes=('_KBO', '_STATIZ'))
        
        # 연도 컬럼 명시
        merged_df['시즌'] = year
        
        all_years_data.append(merged_df)
        print(f"✅ {year}년 병합 완료: 총 {len(merged_df)}명 매칭됨")

    except Exception as e:
        print(f"🚨 {year}년 파일 처리 중 에러 발생: {e}")

# 2. 5개년 데이터를 하나의 거대한 표로 합치기
final_df = pd.concat(all_years_data, ignore_index=True)

# 3. 🚨 노이즈 제거: 스탯티즈 기준 타석(PA)이 50타석 이상인 진짜 선수들만 남기기!
initial_count = len(final_df)
final_df = final_df[final_df['PA_STATIZ'] >= 50].copy()
print(f"\n✂️ 50타석 미만 & 동명이인 오류 필터링: {initial_count}명 -> {len(final_df)}명으로 정제 완료")

# ==========================================
# 🎯 4. 캡스톤 PPT 도메인 지식 (파생 변수) 계산
# ==========================================
print("\n⚙️ 세이버메트릭스 지표(도메인 지식) 계산 중...")

# ① 운의 요소 배제 (BABIP: 인플레이 타구 타율)
denominator_babip = final_df['AB_STATIZ'] - final_df['SO'] - final_df['HR_STATIZ'] + final_df['SF_STATIZ']
final_df['BABIP'] = np.where(denominator_babip > 0, 
                             (final_df['H_STATIZ'] - final_df['HR_STATIZ']) / denominator_babip, 0)

# ② 순수 파워 (ISO: 순수 장타율)
final_df['ISO'] = final_df['SLG'] - final_df['AVG_STATIZ']

# ③ 실질적 득점 가치 (wOBA: 가중 출루율)
final_df['1B'] = final_df['H_STATIZ'] - final_df['2B_STATIZ'] - final_df['3B_STATIZ'] - final_df['HR_STATIZ']
denominator_woba = final_df['AB_STATIZ'] + final_df['BB'] - final_df['IB'] + final_df['SF_STATIZ'] + final_df['HP']
final_df['wOBA'] = np.where(denominator_woba > 0,
                            (0.69 * final_df['BB'] + 0.72 * final_df['HP'] + 0.89 * final_df['1B'] + 
                             1.27 * final_df['2B_STATIZ'] + 1.62 * final_df['3B_STATIZ'] + 2.10 * final_df['HR_STATIZ']) / denominator_woba, 0)

# ④ 고유 능력 / 선구안 (BB/K: 볼넷/삼진 비율)
final_df['BB/K'] = np.where(final_df['SO'] > 0, final_df['BB'] / final_df['SO'], 0)

# 소수점 3자리까지만 깔끔하게 정리
final_df['BABIP'] = final_df['BABIP'].round(3)
final_df['ISO'] = final_df['ISO'].round(3)
final_df['wOBA'] = final_df['wOBA'].round(3)
final_df['BB/K'] = final_df['BB/K'].round(3)

# ==========================================
# 🚨 5. 외국인 타자 일괄 제거 (크레익, 가르시아 추가 완.)
# ==========================================
foreign_hitters = [
    '알테어', '로맥', '피렐라', '마차도', '페르난데스', '터커', '힐리', '페레즈', '크론', 
    '알몬테', '호잉', '보어', '프레이타스', '크레익', '소크라테스', '푸이그', '루이즈', '가르시아', '터크먼', 
    '라가레스', '마티니', '피터스', '렉스', '알포드', '오스틴', '에레디아', '마틴', 
    '도슨', '러셀', '구드럼', '윌리엄스', '오그레디', '데이비슨', '로하스', '페라자', 
    '레이예스', '맥키넌', '라모스', '디아즈', '카데나스', '제러드', '스톤'
]

before_foreigner_drop = len(final_df)
# 이름이 블랙리스트에 '없는(~)' 선수만 남기기 (순수 국내파 필터링)
final_df = final_df[~final_df['이름'].isin(foreign_hitters)]
after_foreigner_drop = len(final_df)

print(f"\n🧹 외국인 타자 정제 완료: {before_foreigner_drop}명 -> {after_foreigner_drop}명 (순수 국내 타자만 남김)")

# ==========================================
# 6. 최종 완성본 파일 저장!
# ==========================================
# 소연님 요청대로 Domestic 제거하고 원본 덮어쓰기로 깔끔하게!
save_path = "/Users/soyeon/Documents/Capstone/data/KBO_Master_Data_2021_2025.csv"
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"🎉 완벽한 5개년 마스터 데이터 생성 성공!!")
print(f"📁 저장 위치: {save_path}")