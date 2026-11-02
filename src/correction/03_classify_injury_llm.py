"""
[3단계] 신뢰도 Low인 건만 Gemini로 재분류 — 투수 kbo_injury_classifier_llm.py와 같은 코드
==========================================================================================
입력 : kbo_injury_batter_v32_rule.csv   (2단계 결과)
출력 : kbo_injury_batter_v32_final.csv

바뀌는 열은 부상분류, 신뢰도, 실제부상여부뿐이다. 심각도점수는 2단계 값을 그대로 둔다.

실행 전: 새로 발급받은 키를 환경변수로 지정 (코드에 키를 적지 말 것)
  export GEMINI_API_KEY="새 키"
"""

import os
import sys
import time
import pandas as pd
from google import genai

DATA_DIR    = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
INPUT_FILE  = os.path.join(DATA_DIR, "kbo_injury_batter_v32_rule.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "kbo_injury_batter_v32_final.csv")

MODEL_NAME = 'gemini-flash-latest'
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("❌ GEMINI_API_KEY 환경변수가 없습니다. 터미널에서 export GEMINI_API_KEY=\"새 키\" 후 실행하세요.")
client = genai.Client(api_key=API_KEY)

# ── 카테고리 → 키워드 매핑 (LLM 응답 파싱용, 한글 키) ──────────────────────
CATEGORY_MAP = {
    '어깨_팔꿈치': ['어깨', '팔꿈치', '손목', '손가락', '손등', '회전근개', '건염',
                    '전완부', '전완', '굴곡근', '이두근', '삼두근', '상완', '회전근', '손톱'],
    '하체_다리':   ['발목', '발가락', '종아리', '무릎', '아킬레스', '허벅지', '비복근',
                    '하지', '햄스트링', '대퇴이두', '대퇴', '내전근', '고관절'],
    '손_허리':     ['허리', '척추', '요추', '디스크', '골반', '광배근',
                    '옆구리', '내복사근', '복근', '복사근', '장요근'],
    '안면_두부':   ['머리', '뇌진탕', '안면', '코뼈', '안구', '헤드샷', '목', '경추'],
    '내과질환':    ['맹장', '장염', '코로나', '백신', '감기', '몸살', '가래톳', '충수염'],
    '기타부상':    ['담 증세', '담증세', '담이', '근육통', '뭉침', '타박상', '골절', '파열', '인대'],
    '성적부진_조정': ['관리', '휴식', '컨디션', '피로', '조정'],
}

# ── 프롬프트 ─────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
다음은 KBO 프로야구 뉴스 기사입니다.

[선수명] {player}
[기사 내용]
{article}

위 기사에서 '{player}' 선수 본인이 부상자 명단(말소)에 오른 이유를 찾아주세요.

답변 규칙:
1. '{player}' 선수 본인의 부상 부위나 질환이 명시되어 있으면 → 부위/질환명만 간단히 (예: 발목, 햄스트링, 허리, 어깨, 담 증세, 골절)
2. '{player}'가 다른 선수 부상 기사에 단순 언급된 경우 → '불명'
3. 부상이 아닌 이유(성적 부진, 전략적 말소, 컨디션 조정)로 보이면 → '비부상'
4. 전혀 알 수 없으면 → '불명'
5. 위 4가지 외 다른 설명은 하지 말고 답변만 출력
"""


# ── LLM 응답 → 한글 카테고리 변환 ──────────────────────────────────────────
def map_to_category(llm_response):
    resp = llm_response.strip()
    if not resp or '불명' in resp:
        return None
    if '비부상' in resp:
        return 'NOT_INJURY'

    matched = []
    for cat, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in resp:
                matched.append((len(kw), cat))

    if matched:
        matched.sort(reverse=True)
        return matched[0][1]
    return None


# ── Gemini API 호출 ──────────────────────────────────────────────────────────
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


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_FILE)

    low_mask = df['신뢰도'] == 'Low'
    low_cnt  = low_mask.sum()

    print(f"[LLM] 입력 파일: {INPUT_FILE}")
    print(f"[LLM] 전체: {len(df)}건 / Low: {low_cnt}건")
    print(f"[LLM] 예상 소요: 약 {low_cnt * 4 // 60}~{low_cnt * 5 // 60}분 (무료 15RPM 기준)\n")

    improved      = 0
    not_injury    = 0
    still_unknown = 0

    for i, (idx, row) in enumerate(df[low_mask].iterrows()):
        player  = str(row['선수']).split('(')[0].strip()
        article = row['뉴스기사제목']

        print(f"[{i+1:3d}/{low_cnt}] {player:5s}", end="  ")

        cat, answer = classify_with_gemini(player, article)

        if cat == 'NOT_INJURY':
            # 비부상 → 성적부진_조정 + 실제부상여부 0
            df.at[idx, '부상분류']     = '성적부진_조정'
            df.at[idx, '신뢰도']       = 'Mid(LLM)'
            df.at[idx, '실제부상여부'] = 0
            not_injury += 1
            print(f"🔵 비부상 → 성적부진_조정 (실제부상=0)")

        elif cat:
            df.at[idx, '부상분류'] = cat
            df.at[idx, '신뢰도']   = 'Mid(LLM)'
            improved += 1
            print(f"✅ '{answer}' → {cat}")

        else:
            # 불명 → 기존 유지
            still_unknown += 1
            print(f"❌ 불명 ('{answer}')")

        # 무료 티어: 15 RPM → 4초 간격
        time.sleep(4)

        # 50건마다 중간 저장
        if (i + 1) % 50 == 0:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"\n  ── 중간 저장 ({i+1}/{low_cnt}건 처리) ──\n")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print('\n' + '=' * 55)
    print(f'완성 → {OUTPUT_FILE}')
    print(f'\nLLM 처리 결과:')
    print(f'  부상 부위 확인:  {improved}건')
    print(f'  비부상 확인:     {not_injury}건')
    print(f'  여전히 불명:     {still_unknown}건')
    print(f'\n최종 신뢰도 분포:')
    print(df['신뢰도'].value_counts().to_string())
    print(f'\n최종 부상분류 분포:')
    print(df['부상분류'].value_counts().to_string())
    print('=' * 55)


if __name__ == '__main__':
    main()