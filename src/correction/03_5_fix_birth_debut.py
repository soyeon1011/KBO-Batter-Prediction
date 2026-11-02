"""
[3.5단계] 출생연도(네이버)·입단연도(나무위키) 다시 수집 — 기존 크롤러의 약점 보완판
=====================================================================================
기존 crawling_age_naver.py / crawling_age_namuwiky.py(투수와 같은 코드)의 약점
  1) 네이버: 화면에 처음 나오는 날짜를 출생연도로 사용 → 동명이인이 먼저 뜨면 틀림 (예: 이정훈)
  2) 나무위키: 이상한 값을 다시 찾다가 못 찾으면 틀린 값을 그대로 둠 (입단연도=출생연도 46명)
  3) 검사 규칙 '입단연도 ≤ 현 소속팀 첫 시즌'이 이적 연도를 걸러내지 못함 (예: 김강민 2024)

이 코드에서 바꾼 점
  1) 네이버: 날짜 바로 앞(60자)에 야구선수·내야수·외야수·포수·KBO가 있을 때만 출생연도로 인정
              (팀명만으로는 인정하지 않음 — 'LG'·'KT'가 들어간 동명 정치인 등 방지)
  2) 나무위키: '이름(출생연도)' 문서를 먼저 확인하고, 문서에 소속팀명이나 'KBO'가 있을 때만 사용
              입단 후보 연도 중 '출생연도+17 ~ +30' 이면서 데이터상 첫 시즌 이전인 값의 가장 이른 연도
              (이적 연도처럼 늦은 연도가 뽑히지 않도록)
  3) 못 찾으면 틀린 값을 남기지 않고 비워 둠

대상 : 입단연도가 비었거나 입단 나이가 17세 미만·24세 초과인 선수만 (약 90명)
       나머지 선수와 은퇴 선수 대부분은 기존 출생연도·입단연도를 그대로 사용
입력 : kbo_final_dataset_methodB.csv   (원본은 수정하지 않음)
출력 : kbo_player_birth_debut_fixed.csv
       이름, 출생연도_기존, 팀목록, 첫시즌, 출생연도_새, 출생_근거, 입단연도_기존, 입단연도_새, 입단_근거
※ 중간에 멈춰도 다시 실행하면 이미 확인한 선수는 건너뛴다. 네이버 차단 시 60초 대기 후 계속.
"""
import os
import random
import re
import time
from urllib.parse import quote, unquote

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

DATA_DIR  = "/Users/soyeon/Documents/GitHub/KBO-Batter-Prediction/data"
BASE_FILE = os.path.join(DATA_DIR, "kbo_final_dataset_methodB.csv")
OUT_FILE  = os.path.join(DATA_DIR, "kbo_player_birth_debut_fixed.csv")

# 04단계와 같은 출생연도 수정 (원본 데이터의 명백한 오류)
BIRTH_FIX = {('박해민', 'LG'): 1990, ('이정훈', 'KT'): 1994}

BASEBALL_WORDS = ['야구선수', '야구 선수', '내야수', '외야수', '포수', 'KBO']   # 팀명만으로는 인정 안 함 (동명 정치인 등 방지)
DATE_RE = re.compile(r'(19[6-9]\d|200\d)\s*[.년]\s*(\d{1,2})\s*[.월]\s*(\d{1,2})')


# ── 출생연도: 네이버 ────────────────────────────────────────────────────────
def find_birth_naver(driver, name, teams):
    for q in [f"{name} 야구선수 {teams[-1]}", f"KBO {teams[-1]} {name}"]:
        driver.get(f"https://search.naver.com/search.naver?query={quote(q)}")
        time.sleep(random.uniform(2.0, 3.5))
        text = driver.find_element(By.TAG_NAME, "body").text
        if "자동입력 방지" in text or "비정상적인 검색" in text:
            print("  🛑 네이버 차단 감지! 60초 대기")
            time.sleep(60)
            continue
        for m in DATE_RE.finditer(text):
            before = text[max(0, m.start() - 60): m.start()]     # 날짜 바로 앞 60자 = 그 사람의 소개
            if name in text[max(0, m.start() - 150): m.start()] and \
                    any(w in before for w in BASEBALL_WORDS):
                return int(m.group(1)), f"naver:{q}"
    return None, None


# ── 입단연도: 나무위키 ──────────────────────────────────────────────────────
ENTRY_PATTERNS = [
    r'프로\s*입단\s*[^\d\n]{0,10}(\d{4})',
    r'(\d{4})년\s*(?:KBO\s*)?(?:신인\s*)?(?:드래프트|2차|1차|육성|신고|자유계약|지명)',
    r'(?:지명\s*순위|입단|데뷔|지명)[^\d\n]{0,15}(\d{4})',
]


def find_debut_namu(driver, name, birth, teams, first_season):
    urls = []
    if birth:
        urls.append(f"https://namu.wiki/w/{quote(f'{name}({birth})')}")
    urls += [f"https://namu.wiki/w/{quote(f'{name}(야구선수)')}"]
    urls += [f"https://namu.wiki/w/{quote(f'{name}({t})')}" for t in dict.fromkeys(teams)]
    urls += [f"https://namu.wiki/w/{quote(name)}"]

    for url in urls:
        try:
            driver.get(url)
            time.sleep(random.uniform(2.0, 3.5))
            text = driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            continue
        if "문서가 존재하지 않습니다" in text:
            continue
        if not (any(t in text for t in teams) or 'KBO' in text):
            continue                                   # 다른 사람 문서
        if birth and str(birth) not in text:
            continue                                   # 출생연도가 안 맞으면 동명이인 문서
        cands = []
        for pat in ENTRY_PATTERNS:
            cands += [int(y) for y in re.findall(pat, text)]
        valid = [y for y in cands
                 if y <= first_season and (birth is None or birth + 17 <= y <= birth + 30)]
        if valid:
            return min(valid), f"namu:{unquote(url.split('/w/')[-1])}"
    return None, None


def main():
    base = pd.read_csv(BASE_FILE)
    for (name, team), b in BIRTH_FIX.items():
        base.loc[(base['이름'] == name) & (base['팀명'] == team), '출생연도'] = b

    players_all = (base.sort_values('연도')
               .groupby(['이름', '출생연도'])
               .agg(팀목록=('팀명', lambda s: list(dict.fromkeys(s))),
                    첫시즌=('연도', 'min'),
                    입단연도_기존=('입단연도', 'last'))
               .reset_index().rename(columns={'출생연도': '출생연도_기존'}))

    # 다시 찾을 대상: 입단연도가 비었거나, 입단 나이가 17세 미만·24세 초과(출생연도와 같음, 이적 연도, 동명이인 등)
    entry_age = players_all['입단연도_기존'] - players_all['출생연도_기존']
    suspect = players_all['입단연도_기존'].isna() | (entry_age < 17) | (entry_age > 24)
    players = players_all[suspect]
    keep = players_all[~suspect]
    print(f"선수 {len(players_all)}명 중 입단연도가 의심되는 {len(players)}명만 다시 찾습니다 "
          f"(나머지 {len(keep)}명은 기존 값 사용)")

    done = pd.read_csv(OUT_FILE) if os.path.exists(OUT_FILE) else pd.DataFrame()
    if len(done):
        done = done[done['출생_근거'] != '기존 값']          # 이전 실행에서 다시 찾은 선수만 이어받기
    done_keys = set(zip(done['이름'], done['출생연도_기존'])) if len(done) else set()
    rows = done.to_dict('records') if len(done) else []
    todo = players[[(n, b) not in done_keys for n, b in zip(players['이름'], players['출생연도_기존'])]]
    print(f"남은 {len(todo)}명\n")

    driver = webdriver.Chrome(options=Options())
    try:
        for i, (_, p) in enumerate(todo.iterrows(), 1):
            name, teams = p['이름'], p['팀목록']
            old_b = int(p['출생연도_기존']) if pd.notna(p['출생연도_기존']) else None
            print(f"[{i:3d}/{len(todo)}] {'/'.join(teams)} {name}", end="  ", flush=True)

            new_b, b_src = find_birth_naver(driver, name, teams)
            birth = new_b or old_b
            new_d, d_src = find_debut_namu(driver, name, birth, teams, int(p['첫시즌']))
            flag = "" if (new_b is None or new_b == old_b) else f"  ⚠ 출생연도 {old_b}→{new_b}"
            print(f"출생 {new_b}  입단 {new_d}{flag}")

            rows.append({'이름': name, '출생연도_기존': p['출생연도_기존'], '팀목록': '/'.join(teams),
                         '첫시즌': p['첫시즌'], '출생연도_새': new_b, '출생_근거': b_src,
                         '입단연도_기존': p['입단연도_기존'], '입단연도_새': new_d, '입단_근거': d_src})
            if i % 10 == 0:
                pd.DataFrame(rows).to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
                print("   💾 저장")
    except KeyboardInterrupt:
        print("\n🛑 중단 - 저장 중...")
    finally:
        found = pd.DataFrame(rows)
        rest = keep.assign(팀목록=keep['팀목록'].map('/'.join), 출생연도_새=keep['출생연도_기존'],
                           출생_근거='기존 값', 입단연도_새=keep['입단연도_기존'], 입단_근거='기존 값')
        out = pd.concat([found, rest], ignore_index=True)
        out.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
        driver.quit()

    print(f"\n💾 {OUT_FILE} ({len(out)}명)")
    print(f"  출생연도 못 찾음: {out['출생연도_새'].isna().sum()}명")
    print(f"  입단연도 못 찾음: {out['입단연도_새'].isna().sum()}명")
    diff = out[out['출생연도_새'].notna() & (out['출생연도_새'] != out['출생연도_기존'])]
    print(f"  출생연도가 기존과 다름: {len(diff)}명  ← 직접 확인 필요")
    if len(diff):
        print(diff[['이름', '팀목록', '출생연도_기존', '출생연도_새', '출생_근거']].to_string(index=False))


if __name__ == '__main__':
    main()
