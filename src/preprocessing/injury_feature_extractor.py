import pandas as pd
import re

# 1. 파일 불러오기 (뉴스 크롤링 완료된 파일)
file_path = "/Users/soyeon/Documents/Capstone/data/kbo_injury_final_with_news.csv"
df = pd.read_csv(file_path)

# 🌟 2. 부상 부위 분류 로직 (소연님의 이전 integrated 파일 기준 9개 카테고리)
def classify_injury(text):
    text = str(text)
    # 어깨/팔꿈치
    if any(w in text for w in ['어깨', '팔꿈치', '엘보', '회전근개']): return '어깨_팔꿈치'
    # 하체/다리
    if any(w in text for w in ['햄스트링', '허벅지', '종아리', '발목', '무릎', '십자인대', '발가락', '내전근']): return '하체_다리'
    # 손/허리 (Trunk & Hand)
    if any(w in text for w in ['허리', '손가락', '손등', '손목', '늑골', '옆구리', '복사뼈', '등']): return '손_허리'
    # 안면/두부
    if any(w in text for w in ['헤드샷', '안면', '코뼈', '턱', '두부', '뇌진탕', '입술']): return '안면_두부'
    # 내과질환
    if any(w in text for w in ['감기', '몸살', '코로나', '장염', '봉와직염', '대상포진', '질환', '컨디션', '피로']): return '내과질환'
    # 사고/충돌
    if any(w in text for w in ['충돌', '사고', '부딪혀', '타구', '맞아', '사구']): return '사고_충돌'
    # 성적부진/조정
    if any(w in text for w in ['조정', '부진', '휴식', '말소', '재정비']): return '성적부진_조정'
    
    return '기타부상'

# 🌟 3. 심각도 점수 로직
def get_severity(text):
    text = str(text)
    if any(w in text for w in ['수술', '파열', '시즌아웃', '완판', '끊어져']): return 5
    if '골절' in text: return 4
    if any(w in text for w in ['손상', '염좌', '미세', '재활', '부분']): return 3
    if any(w in text for w in ['이탈', '통증', '결장', '불편']): return 2
    return 1

print("🔧 [부상 정밀 분석] 뉴스 제목에서 부상 종류를 분류하고 점수를 매깁니다...")

# 데이터 적용
df['부상분류'] = df.apply(lambda x: classify_injury(x.get('뉴스기사제목', x.get('비고', ''))), axis=1)
df['심각도점수'] = df.apply(lambda x: get_severity(x.get('뉴스기사제목', x.get('비고', ''))), axis=1)
df['재검필요'] = df.apply(lambda x: 1 if '재검' in str(x.get('뉴스기사제목', '')) else 0, axis=1)

# 4. 수동 예외 처리 (정수빈, 문상철 등 소연님이 챙기던 선수들)
df.loc[df['선수'] == '정수빈', '재검필요'] = 1
df.loc[df['선수'] == '문상철', '심각도점수'] = 5  # 손등 골절 및 수술 케이스

# 5. 결과 저장
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print(f"✅ 분석 완료! '부상분류'와 '심각도점수'가 {file_path}에 저장되었습니다.")