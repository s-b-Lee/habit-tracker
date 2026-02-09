# app.py
# ─────────────────────────────────────────────────────────────
# AI 습관 트래커 (마법 요정 에디션)
# - "티니핑 감성"을 살린 오리지널(창작) 요정/핑 카드 컨셉
# - 저작권/상표 이슈를 피하기 위해 공식 캐릭터/로고/이미지/고유명사 사용 없음
#
# ✅ 요구 기능 포함
# [기본 설정]
# - 페이지 제목: "AI 습관 트래커 (포켓몬)" -> (요청) "AI 습관 트래커 (마법 요정)"로 변경
# - 아이콘: 🎮
# - 사이드바: OpenAI API Key 입력칸
#
# [습관 체크인 UI]
# - 체크박스 5개 2열 배치 + 이모지
# - 기분 슬라이더 (1~10)
# - 도시 선택 10개 + 코치 스타일 라디오
# - 추가: 물(ml), 운동(분), 메모(주석)
# - 추가: 시간대별(아침/점심/저녁/밤) 체크(시각화용)
#
# [달성률 + 차트]
# - 달성률(%) 계산
# - st.metric 3개: 달성률, 달성 습관, 기분
# - 데모용 6일 + 오늘 데이터로 7일 바 차트
# - session_state로 기록 저장
#
# [API 연동]
# - 날씨 기능 제외(요청)
# - get_fairy_ping(): 랜덤 “핑(요정)” 카드(창작) 생성
#   - 이름/속성/설명/스탯(행복,집중,활력,휴식,용기,반짝)
#
# [AI 코치 리포트]
# - generate_report: 습관+기분+도시+핑 카드 정보를 OpenAI에 전달
# - 코치 스타일별 시스템 프롬프트 (스파르타/멘토/게임마스터)
# - 출력: 컨디션 등급(S~D), 습관 분석, 내일 미션, 오늘의 파트너 핑(스탯 활용 응원)
# - 모델: gpt-5-mini
#
# [결과 표시]
# - '컨디션 리포트 생성' 버튼
# - 2열: (왼쪽) 기록 요약/시각화, (오른쪽) 핑 카드 + 스탯 바 차트(빨간색 요구 → 붉은 계열)
# - AI 리포트
# - 공유용 텍스트 (st.code)
# - 하단 API 안내 (expander)
#
# [추가 요구]
# 1) 캘린더 형태 기록 보기
# 2) 운동/물 등 주석(메모) 달기
# 3) 성공률 시각화: 시간대별/습관종류별 이모지(이미지 느낌)로 표시
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import calendar
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =============================
# 기본 설정
# =============================
st.set_page_config(page_title="AI 습관 트래커 (마법 요정)", page_icon="🎮", layout="wide")

APP_TITLE = "🎮 AI 습관 트래커 (마법 요정 에디션)"
MODEL_NAME = "gpt-5-mini"

HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

TIME_SLOTS = [
    ("🌤️", "아침"),
    ("🏙️", "점심"),
    ("🌆", "저녁"),
    ("🌙", "밤"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Sejong",
    "Jeju",
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]


# =============================
# 유틸
# =============================
def clean(s: str) -> str:
    return (s or "").strip()


def today_iso() -> str:
    return date.today().isoformat()


def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


def iso_to_date(s: str) -> date:
    return date.fromisoformat(s)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return round(n / d * 100, 1)


# =============================
# 오리지널 “핑(요정) 카드” 생성
# =============================
PING_NAMES = [
    "반짝핑",
    "용기핑",
    "집중핑",
    "다정핑",
    "수면핑",
    "정리핑",
    "활력핑",
    "성장핑",
    "미소핑",
    "차분핑",
    "포근핑",
    "신나핑",
]

PING_ELEMENTS = [
    ("💖", "하트"),
    ("✨", "별빛"),
    ("🌿", "초록"),
    ("🌈", "무지개"),
    ("🫧", "버블"),
    ("🎀", "리본"),
]

PING_PHRASES = [
    "오늘은 작은 체크 하나가 마법이 될 거야!",
    "괜찮아, 천천히 해도 돼. 그래도 계속!",
    "너의 리듬을 찾는 중이야. 이미 잘하고 있어.",
    "한 번 반짝이면, 내일은 두 번 반짝!",
    "지금의 너도 충분히 멋져. 다음은 더 좋아져!",
]


def get_fairy_ping(seed_key: Optional[str] = None) -> Dict[str, Any]:
    """
    창작 핑 카드 생성 (API 호출 없이)
    - seed_key가 있으면 같은 날/같은 입력에서 비슷하게 나오도록 결정성 부여 가능
    """
    rng = random.Random(seed_key or f"{today_iso()}-ping")
    name = rng.choice(PING_NAMES)
    emo, element = rng.choice(PING_ELEMENTS)
    phrase = rng.choice(PING_PHRASES)

    # 스탯 (0~100)
    stats = {
        "행복💖": rng.randint(40, 95),
        "집중🌟": rng.randint(30, 95),
        "활력💪": rng.randint(30, 95),
        "휴식💤": rng.randint(30, 95),
        "용기🛡️": rng.randint(30, 95),
        "반짝✨": rng.randint(40, 99),
    }
    return {
        "name": name,
        "element": element,
        "emoji": emo,
        "phrase": phrase,
        "stats": stats,
    }


# =============================
# OpenAI 리포트
# =============================
def _get_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. `pip install openai` 해주세요.")
    return OpenAI(api_key=clean(api_key))


def _style_system_prompt(style: str) -> str:
    base = (
        "너는 사용자의 습관 체크인 데이터를 바탕으로 '코치 리포트'를 작성한다. "
        "의학적/치료적 진단은 하지 말고, 실천 가능한 제안만 한다. "
        "출력 형식을 반드시 지켜라."
    )
    if style == "스파르타 코치":
        return base + " 톤은 엄격하고 직설적이며 짧다. 변명은 끊고 실행 지침을 준다. 모욕 금지."
    if style == "따뜻한 멘토":
        return base + " 톤은 따뜻하고 공감적. 작은 성취를 칭찬하고 부담을 낮춘다."
    return base + " 톤은 RPG/게임마스터처럼. 퀘스트/보상/레벨업 표현으로 재미있게."


def generate_report(
    openai_api_key: str,
    coach_style: str,
    mood: int,
    city: str,
    checked_habits: List[str],
    unchecked_habits: List[str],
    water_ml: int,
    exercise_min: int,
    memo: str,
    time_slots_done: List[str],
    ping: Dict[str, Any],
) -> Optional[str]:
    openai_api_key = clean(openai_api_key)
    if not openai_api_key:
        return None

    ping_text = (
        f"{ping.get('emoji')} {ping.get('name')} ({ping.get('element')})\n"
        f"한마디: {ping.get('phrase')}\n"
        f"스탯: {ping.get('stats')}"
    )

    user_prompt = f"""
아래 데이터를 기반으로 리포트를 작성해줘.

[도시]
{city}

[오늘 기분 점수]
{mood}/10

[완료한 습관]
{", ".join(checked_habits) if checked_habits else "없음"}

[미완료 습관]
{", ".join(unchecked_habits) if unchecked_habits else "없음"}

[물 마시기]
{water_ml} ml

[운동하기]
{exercise_min} 분

[시간대 체크(완료한 시간대)]
{", ".join(time_slots_done) if time_slots_done else "없음"}

[메모(주석)]
{memo if memo else "(없음)"}

[오늘의 파트너 핑(요정 카드)]
{ping_text}

출력 형식(반드시 지켜):
## 컨디션 등급
- 등급: (S/A/B/C/D 중 하나)
- 한 줄 요약: ...

## 습관 분석
- 잘한 점: ...
- 아쉬운 점: ...
- 내일 1% 개선: ...

## 내일 미션
- (실행 미션 3개, 아주 구체적이고 작게)

## 오늘의 파트너 핑
- 핑: (이름/속성)
- 스탯 활용 응원: (스탯 2~3개 끌어와서 오늘의 컨디션에 맞게 응원)
- 한 마디 주문: (짧게 1문장)
""".strip()

    try:
        client = _get_openai_client(openai_api_key)
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": [{"type": "text", "text": _style_system_prompt(coach_style)}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=0.75,
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return str(resp.output_text).strip()

        # fallback
        out_texts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    out_texts.append(getattr(c, "text", ""))
        text = "\n".join([t for t in out_texts if t]).strip()
        return text if text else None
    except Exception:
        return None


# =============================
# 기록(세션) 구조
# =============================
def demo_last_6_days() -> List[Dict[str, Any]]:
    rng = random.Random(20260209)
    today = date.today()
    out: List[Dict[str, Any]] = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        # 습관 체크 수
        checked_cnt = rng.randint(1, 5)
        # 기분
        mood = rng.randint(3, 9)
        # 물/운동
        water = rng.choice([0, 300, 500, 800, 1200, 1500, 2000])
        ex = rng.choice([0, 10, 20, 30, 40, 60, 90])

        # 시간대 체크(랜덤)
        slots = [s for _, s in TIME_SLOTS if rng.random() < 0.5]

        out.append(
            {
                "date": d.isoformat(),
                "habit_checked": checked_cnt,
                "mood": mood,
                "water_ml": water,
                "exercise_min": ex,
                "memo": "",
                "time_slots": slots,
                # 습관별 완료 여부(시각화/캘린더용)
                "habits": {name: (rng.random() < (checked_cnt / 5)) for _, name in HABITS},
            }
        )
    return out


def ensure_state():
    if "records" not in st.session_state:
        st.session_state.records = demo_last_6_days()
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_ping" not in st.session_state:
        st.session_state.last_ping = None


def upsert_today_record(rec: Dict[str, Any]):
    records: List[Dict[str, Any]] = st.session_state.records
    t = today_iso()
    for i, r in enumerate(records):
        if r.get("date") == t:
            records[i] = rec
            break
    else:
        records.append(rec)
    records_sorted = sorted(records, key=lambda x: x.get("date", ""))
    st.session_state.records = records_sorted[-120:]  # 넉넉히 유지(캘린더용)


def get_record_map() -> Dict[str, Dict[str, Any]]:
    return {r["date"]: r for r in st.session_state.records if r.get("date")}


def compute_today_achievement(habits_done: Dict[str, bool]) -> Tuple[int, float]:
    checked_count = sum(1 for v in habits_done.values() if v)
    rate = pct(checked_count, len(HABITS))
    return checked_count, rate


def last_7_days_rate_df() -> pd.DataFrame:
    """
    6일 데모 + 오늘 기록 기반으로 7일 달성률 바 차트용 DF
    """
    recs = sorted(st.session_state.records, key=lambda x: x.get("date", ""))[-7:]
    rows = []
    for r in recs:
        habits = r.get("habits") or {}
        checked = sum(1 for _, name in HABITS if habits.get(name))
        rows.append({"date": r.get("date"), "rate": pct(checked, len(HABITS))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date")
    return df


# =============================
# 캘린더 렌더링
# =============================
def month_calendar_dates(year: int, month: int) -> List[List[Optional[date]]]:
    cal = calendar.Calendar(firstweekday=6)  # 일요일 시작
    weeks = []
    for week in cal.monthdatescalendar(year, month):
        row = []
        for d in week:
            if d.month != month:
                row.append(None)
            else:
                row.append(d)
        weeks.append(row)
    return weeks


def day_badge(rec: Optional[Dict[str, Any]]) -> str:
    """
    캘린더 셀에 표시할 간단 뱃지(이모지):
    - 달성률에 따라 별/하트 느낌으로
    """
    if not rec:
        return "⬜"
    habits = rec.get("habits") or {}
    checked = sum(1 for _, name in HABITS if habits.get(name))
    rate = checked / len(HABITS) if len(HABITS) else 0
    if rate >= 0.8:
        return "💖"
    if rate >= 0.6:
        return "✨"
    if rate >= 0.4:
        return "🫧"
    if rate > 0:
        return "🌧️"
    return "⬜"


# =============================
# 시각화: 시간대별/습관별 성공률(이모지)
# =============================
def slot_success_emoji(p: float) -> str:
    """
    성공률 p(0~1) -> 이모지 게이지
    """
    if p >= 0.85:
        return "🌟🌟🌟🌟🌟"
    if p >= 0.7:
        return "🌟🌟🌟🌟▫️"
    if p >= 0.55:
        return "🌟🌟🌟▫️▫️"
    if p >= 0.35:
        return "🌟🌟▫️▫️▫️"
    if p > 0:
        return "🌟▫️▫️▫️▫️"
    return "▫️▫️▫️▫️▫️"


def habit_success_icon(done: bool, emoji: str) -> str:
    return f"{emoji}✅" if done else f"{emoji}▫️"


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("🔑 OpenAI API Key")
    # (배포 시) secrets 우선값
    default_openai = ""
    try:
        default_openai = str(st.secrets.get("OPENAI_API_KEY", ""))  # type: ignore
    except Exception:
        default_openai = ""
    openai_api_key = st.text_input("OpenAI API Key", value=default_openai, type="password")

    st.divider()
    st.caption("※ 이 에디션은 ‘티니핑 느낌’의 오리지널 요정 컨셉입니다(공식 IP 사용 없음).")


# =============================
# Main
# =============================
ensure_state()

st.title(APP_TITLE)
st.caption("오늘의 작은 습관 체크가 내일의 마법이 돼요 ✨")

# --- 상단 탭 ---
tab1, tab2, tab3 = st.tabs(["✅ 체크인", "🗓️ 캘린더", "📊 시각화"])

# =========================================================
# TAB 1: 체크인
# =========================================================
with tab1:
    st.subheader("✅ 오늘 체크인")

    # 도시 + 코치 스타일
    c0, c1 = st.columns([1, 1])
    with c0:
        city = st.selectbox("🏙️ 도시 선택", options=CITIES, index=0)
    with c1:
        coach_style = st.radio("🧑‍🏫 코치 스타일", options=COACH_STYLES, horizontal=True)

    # 습관 체크박스 2열
    left, right = st.columns(2)
    habits_done: Dict[str, bool] = {}
    for i, (emo, name) in enumerate(HABITS):
        with (left if i % 2 == 0 else right):
            habits_done[name] = st.checkbox(f"{emo} {name}", value=False, key=f"habit_{name}")

    mood = st.slider("😊 오늘 기분 점수", 1, 10, 6)

    # 추가 입력: 물/운동 수치 + 메모
    c2, c3, c4 = st.columns([1, 1, 2])
    with c2:
        water_ml = st.number_input("💧 물 (ml)", min_value=0, max_value=5000, value=500, step=100)
    with c3:
        exercise_min = st.number_input("🏃 운동 (분)", min_value=0, max_value=600, value=20, step=5)
    with c4:
        memo = st.text_input("📝 메모(주석)", value="", placeholder="예: 물 2L 목표! / 하체운동 20분 / 일찍 자기")

    # 시간대 체크(시각화용)
    st.markdown("#### ⏰ 오늘 습관을 주로 실천한 시간대")
    slot_cols = st.columns(4)
    slot_done: Dict[str, bool] = {}
    for i, (emo, slot) in enumerate(TIME_SLOTS):
        with slot_cols[i]:
            slot_done[slot] = st.checkbox(f"{emo} {slot}", value=False, key=f"slot_{slot}")

    # 달성률
    checked_count, rate = compute_today_achievement(habits_done)

    st.markdown("#### 📌 오늘 요약")
    m1, m2, m3 = st.columns(3)
    m1.metric("달성률", f"{rate}%")
    m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
    m3.metric("기분", f"{mood}/10")

    # 저장 버튼 (UI 변경마다 기록이 덮이지 않도록 “저장” 시점 확정)
    st.divider()
    save_col1, save_col2 = st.columns([1, 2])
    with save_col1:
        save = st.button("💾 오늘 기록 저장", type="primary", use_container_width=True)
    with save_col2:
        st.caption("※ 저장을 눌러야 캘린더/통계에 반영됩니다.")

    if save:
        rec = {
            "date": today_iso(),
            "mood": int(mood),
            "water_ml": int(water_ml),
            "exercise_min": int(exercise_min),
            "memo": memo,
            "time_slots": [s for s, v in slot_done.items() if v],
            "habits": habits_done,
        }
        upsert_today_record(rec)
        st.success("오늘 기록이 저장되었어요! ✨")

    # 7일 달성률 차트
    st.subheader("📈 최근 7일 달성률")
    df7 = last_7_days_rate_df()
    if df7.empty:
        st.info("아직 기록이 없어요. 오늘 기록을 저장해보세요!")
    else:
        st.bar_chart(df7.set_index("date")[["rate"]])

    # 리포트 생성: 핑 카드 + AI
    st.subheader("🧠 컨디션 리포트")

    # 오늘의 핑 카드(저장 시점과 무관하게 오늘 기준 고정)
    ping = st.session_state.last_ping or get_fairy_ping(seed_key=today_iso())
    st.session_state.last_ping = ping

    btn = st.button("컨디션 리포트 생성", use_container_width=True)

    if btn:
        if not clean(openai_api_key):
            st.error("OpenAI API Key가 필요해요. 사이드바에 입력해 주세요.")
        else:
            time_slots_done = [s for s, v in slot_done.items() if v]
            report = generate_report(
                openai_api_key=openai_api_key,
                coach_style=coach_style,
                mood=int(mood),
                city=city,
                checked_habits=[k for k, v in habits_done.items() if v],
                unchecked_habits=[k for k, v in habits_done.items() if not v],
                water_ml=int(water_ml),
                exercise_min=int(exercise_min),
                memo=memo,
                time_slots_done=time_slots_done,
                ping=ping,
            )
            st.session_state.last_report = report

    report = st.session_state.last_report

    # 결과 레이아웃(2열): 왼쪽 리포트, 오른쪽 핑 카드
    colL, colR = st.columns([1.2, 1])

    with colR:
        st.markdown("### 🎀 오늘의 파트너 핑")
        st.markdown(f"**{ping['emoji']} {ping['name']}**  ·  *{ping['element']}*")
        st.caption(ping["phrase"])

        # 스탯 바차트 (빨간색 요구 → Altair로 색 지정)
        stats_df = pd.DataFrame({"stat": list(ping["stats"].keys()), "value": list(ping["stats"].values())})

        if alt is not None:
            chart = (
                alt.Chart(stats_df)
                .mark_bar(color="#e74c3c")
                .encode(
                    x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("stat:N", sort="-x"),
                    tooltip=["stat", "value"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            # altair 미설치 시 기본 bar_chart(색 지정 불가)
            st.bar_chart(stats_df.set_index("stat"))

        st.markdown("### 🔗 공유용 텍스트")
        share_payload = {
            "date": today_iso(),
            "city": city,
            "coach_style": coach_style,
            "mood": int(mood),
            "habits": habits_done,
            "water_ml": int(water_ml),
            "exercise_min": int(exercise_min),
            "time_slots": [s for s, v in slot_done.items() if v],
            "memo": memo,
            "ping": ping,
            "report": report,
        }
        st.code(json.dumps(share_payload, ensure_ascii=False, indent=2), language="json")

    with colL:
        st.markdown("### 📝 AI 리포트")
        if report:
            st.markdown(report)
        else:
            st.caption("아직 리포트가 없어요. 버튼을 눌러 생성해보세요.")

        with st.expander("📎 API 안내 / 준비물"):
            st.markdown(
                """
**필요한 것**
- OpenAI API Key (리포트 생성용)

**이 에디션 특징**
- ‘티니핑 느낌’을 살린 **오리지널** 요정(핑) 카드로 구성되어 있어요.
- 공식 캐릭터/로고/이미지는 포함하지 않습니다.

**배포 팁(Streamlit Cloud)**
- Secrets에 `OPENAI_API_KEY` 저장하면 편해요.
"""
            )

# =========================================================
# TAB 2: 캘린더
# =========================================================
with tab2:
    st.subheader("🗓️ 캘린더 기록 보기")

    # 월 선택
    today = date.today()
    cA, cB = st.columns([1, 2])
    with cA:
        year = st.number_input("연도", min_value=2020, max_value=2100, value=today.year, step=1)
        month = st.number_input("월", min_value=1, max_value=12, value=today.month, step=1)

    rec_map = get_record_map()
    weeks = month_calendar_dates(int(year), int(month))

    st.caption("뱃지: 💖(80%↑) ✨(60%↑) 🫧(40%↑) 🌧️(1~39%) ⬜(0%)")

    # 캘린더 그리드
    header = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    cols = st.columns(7)
    for i, h in enumerate(header):
        cols[i].markdown(f"**{h}**")

    for w in weeks:
        cols = st.columns(7)
        for i, d in enumerate(w):
            if d is None:
                cols[i].write(" ")
                continue

            iso = d.isoformat()
            rec = rec_map.get(iso)
            badge = day_badge(rec)

            # 셀 표시
            cols[i].markdown(f"**{d.day}** {badge}")

    st.divider()
    st.markdown("### 🔍 특정 날짜 상세 보기")
    pick = st.date_input("날짜 선택", value=today)
    iso = pick.isoformat()
    rec = rec_map.get(iso)

    if not rec:
        st.info("해당 날짜 기록이 없어요.")
    else:
        habits = rec.get("habits") or {}
        checked = sum(1 for _, name in HABITS if habits.get(name))
        rate = pct(checked, len(HABITS))

        m1, m2, m3 = st.columns(3)
        m1.metric("달성률", f"{rate}%")
        m2.metric("달성 습관", f"{checked}/{len(HABITS)}")
        m3.metric("기분", f"{rec.get('mood', '-')}/10")

        st.markdown("#### ✅ 습관")
        lines = []
        for emo, name in HABITS:
            lines.append(habit_success_icon(bool(habits.get(name)), emo) + f" {name}")
        st.write(" · ".join(lines))

        st.markdown("#### ⏰ 시간대")
        slots = rec.get("time_slots") or []
        st.write(", ".join(slots) if slots else "(없음)")

        st.markdown("#### 💧/🏃 수치")
        st.write(f"- 물: {rec.get('water_ml', 0)} ml")
        st.write(f"- 운동: {rec.get('exercise_min', 0)} 분")

        st.markdown("#### 📝 메모(주석)")
        st.write(rec.get("memo") or "(없음)")

# =========================================================
# TAB 3: 시각화
# =========================================================
with tab3:
    st.subheader("📊 성공률 시각화(이모지)")

    recs = sorted(st.session_state.records, key=lambda x: x.get("date", ""))[-14:]  # 최근 2주 정도로
    if not recs:
        st.info("기록이 없어요. 체크인 탭에서 저장해보세요.")
    else:
        # 1) 습관 종류별 성공률
        st.markdown("### 1) 습관 종류별 성공률 (최근 14일)")
        habit_rates = []
        for emo, name in HABITS:
            total = 0
            done = 0
            for r in recs:
                h = (r.get("habits") or {}).get(name)
                if h is None:
                    continue
                total += 1
                if h:
                    done += 1
            p = (done / total) if total else 0.0
            habit_rates.append((emo, name, p))

        for emo, name, p in habit_rates:
            st.write(f"{emo} **{name}**  ·  {slot_success_emoji(p)}  ({round(p*100,1)}%)")

        st.divider()

        # 2) 시간대별 성공률 (최근 14일) - "그 시간대에 실천했다"고 체크한 비율
        st.markdown("### 2) 시간대별 실천 비율 (최근 14일)")
        slot_rates = []
        for emo, slot in TIME_SLOTS:
            total = len(recs)
            done = 0
            for r in recs:
                slots = r.get("time_slots") or []
                if slot in slots:
                    done += 1
            p = (done / total) if total else 0.0
            slot_rates.append((emo, slot, p))

        for emo, slot, p in slot_rates:
            st.write(f"{emo} **{slot}**  ·  {slot_success_emoji(p)}  ({round(p*100,1)}%)")

        st.divider()

        # 3) 날짜 × 습관 “스티커보드” (이모지로 이미지 느낌)
        st.markdown("### 3) 스티커보드 (날짜 × 습관)")
        st.caption("✅이면 성공 스티커, ▫️이면 빈 칸")

        # 표 형태로 출력(이모지를 활용)
        rows = []
        for r in recs[-10:]:  # 너무 길어지지 않게 최근 10일
            d = r.get("date", "")
            habits = r.get("habits") or {}
            row = {"date": d}
            for emo, name in HABITS:
                row[name] = habit_success_icon(bool(habits.get(name)), emo)
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

        # 4) (선택) 수치 트렌드: 물/운동
        st.markdown("### 4) 물/운동 트렌드 (최근 14일)")
        df2 = pd.DataFrame(
            [
                {
                    "date": r.get("date"),
                    "water_ml": r.get("water_ml", 0),
                    "exercise_min": r.get("exercise_min", 0),
                }
                for r in recs
            ]
        ).sort_values("date")

        cX, cY = st.columns(2)
        with cX:
            st.markdown("#### 💧 물(ml)")
            st.line_chart(df2.set_index("date")[["water_ml"]])
        with cY:
            st.markdown("#### 🏃 운동(분)")
            st.line_chart(df2.set_index("date")[["exercise_min"]])

st.caption("© AI 습관 트래커 (마법 요정 에디션) — 오늘의 체크가 내일의 마법 ✨")
