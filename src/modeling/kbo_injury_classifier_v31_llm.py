"""
KBO 부상 분류 v31 — Gemini LLM Fallback
=========================================
v30에서 신뢰도 Low인 케이스를 Gemini API로 재분류.
Rule-based(v30) + LLM(Gemini) Hybrid 파이프라인.

사용 전 준비:
  pip install google-generativeai
"""

import os
import time
import pandas as pd
from google import genai

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyA2KE4q6ouRTHYmpU42O7sVdOVAx1BhdXk")

DATA_DIR     = "/Users/soyeon/Documents/Capstone/data"
INPUT_FILE   = os.path.join(DATA_DIR, "kbo_injury_final_classified_v30_master.csv")
OUTPUT_FILE  = os.path.join(DATA_DIR, "kbo_injury_final_classified_v31_master.csv")

# Gemini 클라이언트 설정 (신규 패키지)
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = 'gemini-flash-latest'


# ─────────────────────────────────────────────────────────────────────────────
# 카테고리 → 키워드 매핑 (LLM 응답 파싱용)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    'Leg_Foot':      ['발목', '발가락', '종아리', '무릎', '아킬레스', '허벅지', '비복근', '하지'],
    'Arm_Shoulder':  ['어깨', '팔꿈치', '손목', '손가락', '손등', '회전근개', '건염', '상지'],
    'Side':          ['옆구리', '내복사근', '복근', '복사근', '고관절', '장요근', '엉덩이'],
    'Hamstring':     ['햄스트링', '대퇴이두', '대퇴', '내전근'],
    'Back':          ['허리', '척추', '요추', '디스크', '골반'],
    'Neck':          ['목', '경추'],
    'Head_Eye':      ['머리', '뇌진탕', '안면', '코뼈', '안구', '헤드샷'],
    'Stiffness':     ['담 증세', '담증세', '담이', '근육통', '뭉침', '타박상'],
    'Disease':       ['맹장', '장염', '코로나', '백신', '감기', '몸살'],
    'Minor_Management': ['관리', '휴식', '컨디션', '피로'],
}

# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
다음은 KBO 프로야구 뉴스 기사입니다.

[선수명] {player}
[기사 내용]
{article}

위 기사에서 '{player}' 선수 본인이 부상자 명단(말소)에 오른 이유를 찾아주세요.

답변 규칙:
1. '{player}' 선수 본인의 부상 부위나 질환이 명시되어 있으면 → 부위/질환명만 간단히 (예: 발목, 햄스트링, 허리, 어깨, 담 증세)
2. '{player}'가 다른 선수 부상 기사에 단순 언급된 경우 → '불명'
3. 부상이 아닌 이유(성적 부진, 전략적 말소)로 보이면 → '비부상'
4. 전혀 알 수 없으면 → '불명'
5. 위 4가지 외 다른 설명은 하지 말고 답변만 출력
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM 응답 → 카테고리 변환
# ─────────────────────────────────────────────────────────────────────────────
def map_to_category(llm_response):
    resp = llm_response.strip()

    # 분류 불가
    if not resp or '불명' in resp:
        return None

    # 비부상으로 판단 → Other_Injury로 처리 (부상발생여부도 0으로 바꿀 수 있음)
    if '비부상' in resp:
        return 'NOT_INJURY'

    # 카테고리 키워드 매칭 (긴 키워드 우선)
    matched = []
    for cat, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in resp:
                matched.append((len(kw), cat))

    if matched:
        matched.sort(reverse=True)
        return matched[0][1]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API 호출 (재시도 포함)
# ─────────────────────────────────────────────────────────────────────────────
def classify_with_gemini(player, article_text, retry=2):
    text = str(article_text)[:1000] if pd.notna(article_text) else ''
    if not text or text in ['관련 기사 없음', '검색 결과 없음']:
        return None, '기사없음'

    prompt = PROMPT_TEMPLATE.format(player=player, article=text)

    for attempt in range(retry + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            answer = response.text.strip()
            cat    = map_to_category(answer)
            return cat, answer
        except Exception as e:
            print(f"  ⚠ API 오류({attempt+1}): {e}")
            if attempt < retry:
                time.sleep(10)

    return None, '오류'


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df       = pd.read_csv(INPUT_FILE)
    low_mask = df['신뢰도'] == 'Low'
    low_cnt  = low_mask.sum()

    print(f"[v31] v30 로드: {len(df)}건")
    print(f"[v31] Low 케이스: {low_cnt}건 → Gemini 재분류 시작")
    print(f"[v31] 예상 소요: 약 {low_cnt * 4 // 60}~{low_cnt * 5 // 60}분\n")

    improved     = 0
    not_injury   = 0
    still_unknown = 0

    for i, (idx, row) in enumerate(df[low_mask].iterrows()):
        player  = str(row['선수']).split('(')[0].strip()
        article = row['뉴스기사제목']

        print(f"[{i+1:3d}/{low_cnt}] {player:5s}", end="  ")

        cat, answer = classify_with_gemini(player, article)

        if cat == 'NOT_INJURY':
            # 비부상 → Other_Injury + 실제부상여부 0
            df.at[idx, '부상분류']    = 'Other_Injury'
            df.at[idx, '신뢰도']      = 'Mid(LLM)'
            df.at[idx, '실제부상여부'] = 0
            not_injury += 1
            print(f"🔵 비부상 확인 → Other_Injury(실제부상=0)")

        elif cat:
            df.at[idx, '부상분류'] = cat
            df.at[idx, '신뢰도']   = 'Mid(LLM)'
            improved += 1
            print(f"✅ '{answer}' → {cat}")

        else:
            still_unknown += 1
            print(f"❌ '{answer}'")

        # Rate limit 준수 (무료 티어: 15 RPM → 4초 간격)
        time.sleep(1)

        # 50건마다 중간 저장
        if (i + 1) % 50 == 0:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"\n  ── 중간 저장 ({i+1}/{low_cnt}건 처리) ──\n")

    # 최종 저장
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    # 결과 요약
    print('\n' + '=' * 55)
    print(f'v31 완성 → {OUTPUT_FILE}')
    print(f'\nLLM 처리 결과:')
    print(f'  부상 부위 확인:  {improved}건')
    print(f'  비부상 확인:     {not_injury}건')
    print(f'  여전히 불명:     {still_unknown}건')
    print(f'\n신뢰도 분포:')
    print(df['신뢰도'].value_counts().to_string())
    print(f'\n부상분류 분포:')
    print(df['부상분류'].value_counts().to_string())
    print('=' * 55)


if __name__ == '__main__':
    main()