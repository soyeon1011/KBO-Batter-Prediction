"""
[2단계] 규칙 기반 부상 분류 — 투수 코드와 같은 규칙
====================================================
입력 : kbo_injury_batter_v32_news_all.csv   (1-2단계 결과, 1,297건)
출력 : kbo_injury_batter_v32_rule.csv

투수 파이프라인을 그대로 적용
  (1) injury_feature_extractor.py     : 기사 → 부상분류(8개), 심각도점수(키워드 1~5점), 재검필요
  (2) kbo_pitcher_injury_classifier.py: 오분류 보정, 신뢰도(High/Mid/Low), 실제부상여부
타자 데이터에 맞춘 차이 1가지
  - 2군 출장이 없을 때 투수 데이터는 'X', 타자 데이터는 '0경기'로 기록되어 있어 이를 같은 뜻으로 처리
"""
import os
import re
import pandas as pd

DATA_DIR    = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
INPUT_FILE  = os.path.join(DATA_DIR, "kbo_injury_batter_v32_news_all.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "kbo_injury_batter_v32_rule.csv")
NO_FUTURES  = ['X', '0경기']        # 부상 기간 중 2군 출장 없음


# ── (1) 투수 injury_feature_extractor.py 함수 그대로 ─────────────────────────
def classify_injury(text):
    text = str(text)

    # '등' 오인식 방지: "A 등 B" 형태는 신체 부위가 아님
    back_keyword = '등' in text
    if back_keyword:
        if re.search(r'등[이을은도가와과]', text) or re.search(r'[가-힣] 등 [가-힣]', text):
            back_keyword = False

    # 어깨/팔꿈치 (투수 핵심 부위)
    if any(w in text for w in ['어깨', '팔꿈치', '엘보', '회전근개', '건염', '건초염',
                                '전완부', '전완', '굴곡근', '이두근', '삼두근', '상완',
                                '회전근', '상지', '손톱',
                                '팔뚝']): return '어깨_팔꿈치' 

    # 하체/다리
    if any(w in text for w in ['햄스트링', '허벅지', '종아리', '발목', '무릎', '십자인대',
                                '발가락', '내전근', '고관절', '비복근', '아킬레스',
                                '대퇴', '하지']): return '하체_다리'

    # 손/허리
    if any(w in text for w in ['허리', '손가락', '손등', '손목', '늑골', '갈비뼈', '옆구리',
                                '복사뼈', '내복사근', '복근', '복사근', '척추', '요추',
                                '디스크', '골반', '광배근', '활배근',
                                '장요근']) or back_keyword: return '손_허리'

    # 안면/두부
    if any(w in text for w in ['헤드샷', '안면', '코뼈', '턱', '두부', '뇌진탕',
                                '입술', '목', '경추']): return '안면_두부'

    # 내과질환
    if any(w in text for w in ['감기', '몸살', '코로나', '장염', '봉와직염', '대상포진',
                                '맹장', '충수염', '가래톳', '백신', '질환']): return '내과질환'

    # 사고/충돌
    if any(w in text for w in ['충돌', '사고', '부딪혀', '타구', '맞아', '사구']): return '사고_충돌'

    # 성적부진/조정
    if any(w in text for w in ['조정', '부진', '휴식', '재정비', '컨디션', '피로']): return '성적부진_조정'

    return '기타부상'

# 🌟 3. 심각도 점수 로직
def get_severity(text):
    text = str(text)
    if any(w in text for w in ['수술', '파열', '시즌아웃', '끊어져']): return 5
    if '골절' in text: return 4
    if any(w in text for w in ['손상', '염좌', '미세', '재활', '부분']): return 3
    if any(w in text for w in ['이탈', '통증', '결장', '불편']): return 2
    return 1


# ── (2) 투수 kbo_pitcher_injury_classifier.py 함수 그대로 ────────────────────
def fix_classification(row):
    news = str(row['뉴스기사제목'])
    분류 = row['부상분류']

    # 1. 피로골절 / 스트레스골절 → 성적부진_조정 오분류 수정
    #    "피로" 키워드에 걸렸지만 실제로는 골절 부상
    if 분류 == '성적부진_조정' and re.search(r'피로골절|스트레스\s*골절', news):
        return '기타부상'

    # 2. 볼넷(사구) → 사고_충돌 오분류 수정
    #    "사구가 많았다" 등 볼넷 맥락인데 몸에 맞는 공(HBP) 맥락이 없으면 수정
    if 분류 == '사고_충돌':
        is_bolnnet    = bool(re.search(r'사구(?:가|를|는|도|로|만)', news))
        is_hitbypitch = bool(re.search(r'몸에\s*맞|데드볼|사구\s*맞|맞은\s*공|타구\s*맞', news))
        if is_bolnnet and not is_hitbypitch:
            return '기타부상'

    return 분류


def assign_confidence(row):
    news = str(row['뉴스기사제목'])
    분류 = row['부상분류']

    if news == '관련 기사 없음':
        return 'Low'
    if 분류 == '기타부상':
        return 'Low'
    if 분류 == '성적부진_조정':
        return 'Mid'
    if 분류 in ['어깨_팔꿈치', '하체_다리', '손_허리', '안면_두부', '사고_충돌']:
        return 'High'
    if 분류 == '내과질환':
        return 'Mid'
    return 'Low'


def assign_real_injury(row):
    """투수 코드와 같은 규칙: 성적부진_조정 → 0, 기사 없음 + 2군 출장 있음 → 0"""
    if row['부상분류'] == '성적부진_조정':
        return 0
    futures = str(row['2군출장상세']).strip()
    if row['뉴스기사제목'] == '관련 기사 없음' and futures not in NO_FUTURES and futures != 'nan':
        return 0
    return 1


def main():
    df = pd.read_csv(INPUT_FILE)
    news = df['뉴스기사제목'].fillna('').astype(str).str.strip()
    df['뉴스기사제목'] = news.where(news != '', '관련 기사 없음')

    df['부상분류']   = df['뉴스기사제목'].apply(classify_injury)
    df['심각도점수'] = df['뉴스기사제목'].apply(get_severity)
    df['재검필요']   = df['뉴스기사제목'].apply(lambda x: 1 if '재검' in str(x) else 0)

    df['부상분류']     = df.apply(fix_classification, axis=1)
    df['신뢰도']       = df.apply(assign_confidence, axis=1)
    df['실제부상여부'] = df.apply(assign_real_injury, axis=1)

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ {len(df)}건 → {OUTPUT_FILE}")
    print("\n[심각도점수]");  print(df['심각도점수'].value_counts().sort_index().to_string())
    print("\n[부상분류]");    print(df['부상분류'].value_counts().to_string())
    print("\n[신뢰도]");      print(df['신뢰도'].value_counts().to_string())
    print("\n[실제부상여부]"); print(df['실제부상여부'].value_counts().to_string())

    # 검토용: 심각도 4~5점 (다른 선수 기사가 섞였는지 눈으로 확인)
    check = df[df['심각도점수'] >= 4][['팀', '선수', '부상시작일', '실제결장일수', '심각도점수', '부상분류', '뉴스기사제목']]
    path = os.path.join(DATA_DIR, "kbo_injury_batter_v32_check_severity45.csv")
    check.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n검토용 심각도 4~5점 {len(check)}건 → {path}")


if __name__ == '__main__':
    main()
