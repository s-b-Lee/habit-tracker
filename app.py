# app.py
# ─────────────────────────────────────────────────────────────
# AI 습관 트래커 (마법 요정 에디션) - 오류 수정 + 지역 선택 제거 + 캘린더 직관 강화
#
# ✅ 이번 수정 포인트(사용자 에러 로그 반영)
# 1) Responses API 오류(400):
#    - content type을 'text' -> 'input_text'로 수정
# 2) Chat Completions 오류(400):
#    - gpt-5-mini 모델에서 temperature=0.75 미지원 → temperature 파라미터 제거(기본값 사용)
# 3) "지역(도시) 선택" UI 제거
# 4) 캘린더 형태로 습관 트래커를 더 직관적으로:
#    - 월간 캘린더(7열) + 날짜칸에 습관 스티커(이모지✅/▫️) 표시
#    - 날짜 선택 → 그날 기록 편집/저장
#
# ✅ 포함 기능
# - 사이드바: OpenAI API Key (secrets 우선)
# - 습관 체크인: 5개(2열), 기분(1~10), 코치 스타일(3),
#               물(ml), 운동(분), 메모, 시간대 체크
# - 최근 7일 달성률 바 차트
# - 오리지널 “파트너 핑(요정 카드)” + 스탯 바 차트(빨간색)
# - AI 컨디션 리포트(모델 gpt-5-mini)
# - 공유용 JSON
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import calendar
import json
import random
from datetime import date, timedelta
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

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]


# =============================
# 유틸
# =============================
def clean(s: str) -> str:
    return (s or "").strip()


def iso(d: date) -> str:
    return d.isoformat()


def pct(n: int, d: int) -> float:
    return round((n / d * 100) if d else 0.0, 1)


def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def calc_checked(habits: Dict[str, bool]) -> int:
    return sum(1 for _, name in HABITS if habits.get(name))


def ensure_habits_dict(h: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    base = {name: False for _, name in HABITS}
    if not h:
        return base
    for _, name in HABITS:
        base[name] = bool(h.get(name))
    return base


# =============================
# 오리지널 “핑(요정) 카드”
# =============================
PING_NAMES = [
    "반짝핑", "용기핑", "집중핑", "다정핑", "수면핑", "정리핑",
    "활력핑", "성장핑", "미소핑", "차분핑", "포근핑", "신나핑"
]
PING_ELEMENTS = [("💖", "하트"), ("✨", "별빛"), ("🌿", "초록"), ("🌈", "무지개"), ("🫧", "버블"), ("🎀", "리본")]
PING_PHRASES = [
    "오늘은 작은 체크 하나가 마법이 될 거야!",
    "괜찮아, 천천히 해도 돼. 그래도 계속!",
    "너의 리듬을 찾는 중이야. 이미 잘하고 있어.",
    "한 번 반짝이면, 내일은 두 번 반짝!",
    "지금의 너도 충분히 멋져. 다음은 더 좋아져!",
]


def get_fairy_ping(seed_key: str) -> Dict[str, Any]:
    rng = random.Random(seed_key)
    name = rng.choice(PING_NAMES)
    emo, element = rng.choice(PING_ELEMENTS)
    phrase = rng.choice(PING_PHRASES)
    stats = {
        "행복💖": rng.randint(40, 95),
        "집중🌟": rng.randint(30, 95),
        "활력💪": rng.randint(30, 95),
        "휴식💤": rng.randint(30, 95),
        "용기🛡️": rng.randint(30, 95),
        "반짝✨": rng.randint(40, 99),
    }
    return {"name": name, "emoji": emo, "element": element, "phrase": phrase, "stats": stats}


# =============================
# OpenAI 리포트 (오류 수정/안정화)
# =============================
def _get_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 없습니다. requirements.txt에 openai를 추가하고 재실행하세요.")
    return OpenAI(api_key=clean(api_key))


def _style_system_prompt(style: str) -> str:
    base = (
        "너는 사용자의 습관 체크인 데이터를 바탕으로 '코치 리포트'를 작성한다. "
        "의학적/치료적 진단은 하지 말고, 실천 가능한 제안만 한다. "
        "출력 형식을 반드시 지켜라."
    )
    if style == "스파르타 코치":
        return base + " 톤은 엄격하고 직설적이며 짧다. 변명은 끊고 실행 지침만 준다. 모욕 금지."
    if style == "따뜻한 멘토":
        return base + " 톤은 따뜻하고 공감적. 작은 성취를 칭찬하고 부담을 낮춘다."
    return base + " 톤은 RPG/게임마스터처럼. 퀘스트/보상/레벨업 표현으로 재미있게."


def build_user_prompt(
    mood: int,
    checked_habits: List[str],
    unchecked_habits: List[str],
    water_ml: int,
    exercise_min: int,
    memo: str,
    time_slots_done: List[str],
    ping: Dict[str, Any],
) -> str:
    ping_text = (
        f"{ping.get('emoji')} {ping.get('name')} ({ping.get('element')})\n"
        f"한마디: {ping.get('phrase')}\n"
        f"스탯: {ping.get('stats')}"
    )
    return f"""
아래 데이터를 기반으로 리포트를 작성해줘.

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
- 스탯 활용 응원: (스탯 2~3개 끌어와서 응원)
- 한 마디 주문: (짧게 1문장)
""".strip()


def generate_report(
    api_key: str,
    coach_style: str,
    user_prompt: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: (report_text_or_None, error_message_or_None)
    - Responses API 우선
    - 실패 시 Chat Completions 폴백
    - gpt-5-mini에서 temperature 커스텀 미지원 -> temperature 파라미터 제거
    """
    api_key = clean(api_key)
    if not api_key:
        return None, "OpenAI API Key가 비어있습니다."

    client = _get_openai_client(api_key)

    # 1) Responses API (content type: input_text)
    try:
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": _style_system_prompt(coach_style)}],
                },
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            # temperature 미지정(기본값 사용)
        )
        if getattr(resp, "output_text", None):
            return str(resp.output_text).strip(), None

        # fallback extraction
        out_texts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "summary_text"):
                    out_texts.append(getattr(c, "text", ""))
        text = "\n".join([t for t in out_texts if t]).strip()
        if text:
            return text, None

        return None, "Responses API 응답이 비어있습니다."
    except Exception as e_responses:
        last_err = f"Responses API 실패: {type(e_responses).__name__}: {e_responses}"

    # 2) Chat Completions fallback (temperature 제거)
    try:
        cc = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _style_system_prompt(coach_style)},
                {"role": "user", "content": user_prompt},
            ],
            # temperature 미지정(기본값 사용)
        )
        content = cc.choices[0].message.content if cc and cc.choices else None
        if content:
            return content.strip(), None
        return None, "Chat Completions 응답이 비어있습니다."
    except Exception as e_chat:
        return None, (last_err + "\n" + f"Chat Completions 실패: {type(e_chat).__name__}: {e_chat}").strip()


# =============================
# 기록 저장 (session_state)
# =============================
def demo_last_6_days() -> List[Dict[str, Any]]:
    rng = random.Random(20260209)
    today = date.today()
    out = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked_cnt = rng.randint(1, 5)
        mood = rng.randint(3, 9)
        water = rng.choice([0, 300, 500, 800, 1200, 1500, 2000])
        ex = rng.choice([0, 10, 20, 30, 40, 60, 90])
        slots = [s for _, s in TIME_SLOTS if rng.random() < 0.5]

        # 정확히 checked_cnt 개수만 True가 되도록 샘플 생성
        habit_names = [name for _, name in HABITS]
        trues = set(rng.sample(habit_names, k=checked_cnt))
        habits = {name: (name in trues) for name in habit_names}

        out.append(
            {
                "date": iso(d),
                "mood": mood,
                "water_ml": water,
                "exercise_min": ex,
                "memo": "",
                "time_slots": slots,
                "habits": habits,
            }
        )
    return out


def ensure_state():
    if "records" not in st.session_state:
        st.session_state.records = demo_last_6_days()
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = date.today()
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_openai_error" not in st.session_state:
        st.session_state.last_openai_error = None


def rec_map() -> Dict[str, Dict[str, Any]]:
    return {r["date"]: r for r in st.session_state.records if r.get("date")}


def get_rec(d: date) -> Optional[Dict[str, Any]]:
    return rec_map().get(iso(d))


def upsert_rec(rec: Dict[str, Any]):
    records: List[Dict[str, Any]] = st.session_state.records
    key = rec["date"]
    for i, r in enumerate(records):
        if r.get("date") == key:
            records[i] = rec
            break
    else:
        records.append(rec)
    st.session_state.records = sorted(records, key=lambda x: x.get("date", ""))[-365:]


def last_7_days_rate_df() -> pd.DataFrame:
    recs = sorted(st.session_state.records, key=lambda x: x.get("date", ""))[-7:]
    rows = []
    for r in recs:
        habits = ensure_habits_dict(r.get("habits"))
        checked = calc_checked(habits)
        rows.append({"date": r.get("date"), "rate": pct(checked, len(HABITS))})
    df = pd.DataFrame(rows)
    return df.sort_values("date") if not df.empty else df


# =============================
# 캘린더 UI helpers
# =============================
def month_grid(year: int, month: int) -> List[List[Optional[date]]]:
    cal = calendar.Calendar(firstweekday=6)  # Sunday
    weeks: List[List[Optional[date]]] = []
    for week in cal.monthdatescalendar(year, month):
        row: List[Optional[date]] = []
        for d in week:
            row.append(d if d.month == month else None)
        weeks.append(row)
    return weeks


def badge_from_rate(rate: float) -> str:
    if rate >= 80:
        return "💖"
    if rate >= 60:
        return "✨"
    if rate >= 40:
        return "🫧"
    if rate > 0:
        return "🌧️"
    return "⬜"


def cell_stickers(habits: Dict[str, bool]) -> str:
    parts = []
    for emo, name in HABITS:
        parts.append(f"{emo}{'✅' if habits.get(name) else '▫️'}")
    return " ".join(parts)


# =============================
# Sidebar
# =============================
ensure_state()

with st.sidebar:
    st.header("🔑 OpenAI API Key")
    default_openai = ""
    try:
        default_openai = str(st.secrets.get("OPENAI_API_KEY", ""))  # type: ignore
    except Exception:
        default_openai = ""
    openai_api_key = st.text_input("OpenAI API Key", value=default_openai, type="password")

    st.divider()
    st.caption("이 에디션은 ‘티니핑 느낌’의 **오리지널** 요정 컨셉입니다(공식 IP 사용 없음).")


# =============================
# Main
# =============================
st.title(APP_TITLE)
st.caption("월간 캘린더에서 스티커처럼 습관을 한눈에 확인하고, AI 리포트로 내일을 준비해요 ✨")

today = date.today()
sel: date = st.session_state.selected_day

# 월/날짜 선택
c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1.2])
with c0:
    year = st.number_input("연도", min_value=2020, max_value=2100, value=sel.year, step=1)
with c1:
    month = st.number_input("월", min_value=1, max_value=12, value=sel.month, step=1)
with c2:
    if st.button("오늘로 이동"):
        st.session_state.selected_day = today
        sel = today
with c3:
    picked = st.date_input("선택 날짜", value=sel)
    st.session_state.selected_day = picked
    sel = picked

st.divider()

# ── 캘린더 그리드
st.subheader("🗓️ 월간 습관 캘린더")
st.caption("뱃지: 💖(80%↑) ✨(60%↑) 🫧(40%↑) 🌧️(1~39%) ⬜(0%)  ·  스티커: 이모지✅/▫️")

grid = month_grid(int(year), int(month))
rmap = rec_map()

headers = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
hcols = st.columns(7)
for i, h in enumerate(headers):
    hcols[i].markdown(f"**{h}**")

for week in grid:
    cols = st.columns(7)
    for i, d in enumerate(week):
        if d is None:
            cols[i].write(" ")
            continue

        rec = rmap.get(iso(d))
        habits = ensure_habits_dict(rec.get("habits") if rec else None)
        checked = calc_checked(habits)
        rate = pct(checked, len(HABITS))
        badge = badge_from_rate(rate)

        title = f"**{d.day}** {badge}" + ("  ✅" if d == sel else "")
        cols[i].markdown(title)
        cols[i].caption(cell_stickers(habits))

        if cols[i].button("선택", key=f"pick_{iso(d)}"):
            st.session_state.selected_day = d
            st.rerun()

st.divider()

# ── 선택 날짜 기록 편집
st.subheader(f"✍️ 기록 입력/수정 — {sel.isoformat()}")

existing = get_rec(sel)
default_habits = ensure_habits_dict(existing.get("habits") if existing else None)
default_mood = safe_int(existing.get("mood"), 6) if existing else 6
default_water = safe_int(existing.get("water_ml"), 500) if existing else 500
default_ex = safe_int(existing.get("exercise_min"), 20) if existing else 20
default_memo = str(existing.get("memo") or "") if existing else ""
default_slots = set(existing.get("time_slots") or []) if existing else set()

coach_style = st.radio("🧑‍🏫 코치 스타일", options=COACH_STYLES, horizontal=True, key="coach_style")

lcol, rcol = st.columns(2)
habits_done: Dict[str, bool] = {}
for idx, (emo, name) in enumerate(HABITS):
    with (lcol if idx % 2 == 0 else rcol):
        habits_done[name] = st.checkbox(f"{emo} {name}", value=bool(default_habits.get(name)), key=f"habit_{sel}_{name}")

mood = st.slider("😊 기분 점수", 1, 10, default_mood, key=f"mood_{sel}")

cA, cB, cC = st.columns([1, 1, 2])
with cA:
    water_ml = st.number_input("💧 물 (ml)", min_value=0, max_value=5000, value=default_water, step=100, key=f"water_{sel}")
with cB:
    exercise_min = st.number_input("🏃 운동 (분)", min_value=0, max_value=600, value=default_ex, step=5, key=f"ex_{sel}")
with cC:
    memo = st.text_input("📝 메모(주석)", value=default_memo, placeholder="예: 물 2L 목표 / 하체운동 / 일찍 자기", key=f"memo_{sel}")

st.markdown("#### ⏰ 실천 시간대(체크)")
slot_cols = st.columns(4)
slot_done: Dict[str, bool] = {}
for i, (emo, slot) in enumerate(TIME_SLOTS):
    with slot_cols[i]:
        slot_done[slot] = st.checkbox(f"{emo} {slot}", value=(slot in default_slots), key=f"slot_{sel}_{slot}")

checked_count = calc_checked(habits_done)
rate = pct(checked_count, len(HABITS))

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{rate}%")
m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")

save1, save2 = st.columns([1, 2])
with save1:
    save_btn = st.button("💾 저장", type="primary", use_container_width=True)
with save2:
    st.caption("저장하면 캘린더/통계/리포트에 반영됩니다.")

if save_btn:
    rec = {
        "date": iso(sel),
        "mood": int(mood),
        "water_ml": int(water_ml),
        "exercise_min": int(exercise_min),
        "memo": memo,
        "time_slots": [s for s, v in slot_done.items() if v],
        "habits": habits_done,
    }
    upsert_rec(rec)
    st.success("저장 완료! 캘린더가 업데이트됩니다.")
    st.rerun()

st.divider()

# ── 최근 7일 차트
st.subheader("📈 최근 7일 달성률")
df7 = last_7_days_rate_df()
if df7.empty:
    st.info("아직 기록이 없어요.")
else:
    st.bar_chart(df7.set_index("date")[["rate"]])

st.divider()

# ── 리포트 + 핑 카드
st.subheader("🧠 컨디션 리포트 & 오늘의 파트너 핑")

ping = get_fairy_ping(seed_key=f"{iso(sel)}-ping")
stats_df = pd.DataFrame({"stat": list(ping["stats"].keys()), "value": list(ping["stats"].values())})

gen = st.button("컨디션 리포트 생성", use_container_width=True)

if gen:
    if not clean(openai_api_key):
        st.session_state.last_report = None
        st.session_state.last_openai_error = "OpenAI API Key가 비어있습니다. 사이드바에 입력해 주세요."
    else:
        user_prompt = build_user_prompt(
            mood=int(mood),
            checked_habits=[k for k, v in habits_done.items() if v],
            unchecked_habits=[k for k, v in habits_done.items() if not v],
            water_ml=int(water_ml),
            exercise_min=int(exercise_min),
            memo=memo,
            time_slots_done=[s for s, v in slot_done.items() if v],
            ping=ping,
        )
        with st.spinner("AI가 리포트를 작성하는 중..."):
            report, err = generate_report(openai_api_key, coach_style, user_prompt)
        st.session_state.last_report = report
        st.session_state.last_openai_error = err

colL, colR = st.columns([1.2, 1])

with colR:
    st.markdown("### 🎀 파트너 핑 카드")
    st.markdown(f"**{ping['emoji']} {ping['name']}**  ·  *{ping['element']}*")
    st.caption(ping["phrase"])

    # 빨간색 바 차트
    if alt is not None:
        chart = (
            alt.Chart(stats_df)
            .mark_bar(color="#e74c3c")
            .encode(
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("stat:N", sort="-x"),
                tooltip=["stat", "value"],
            )
            .properties(height=230)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.bar_chart(stats_df.set_index("stat"))

    st.markdown("### 🔗 공유용 텍스트")
    share = {
        "date": iso(sel),
        "coach_style": coach_style,
        "mood": int(mood),
        "habits": habits_done,
        "water_ml": int(water_ml),
        "exercise_min": int(exercise_min),
        "time_slots": [s for s, v in slot_done.items() if v],
        "memo": memo,
        "ping": ping,
        "report": st.session_state.last_report,
        "openai_error": st.session_state.last_openai_error,
    }
    st.code(json.dumps(share, ensure_ascii=False, indent=2), language="json")

with colL:
    st.markdown("### 📝 AI 리포트")
    if st.session_state.last_report:
        st.markdown(st.session_state.last_report)
    else:
        st.caption("아직 리포트가 없어요. 버튼을 눌러 생성해보세요.")

    if st.session_state.last_openai_error:
        st.error("리포트 생성 오류가 발생했어요.")
        with st.expander("오류 상세 보기"):
            st.code(st.session_state.last_openai_error)

    with st.expander("📎 API 안내 / 준비물"):
        st.markdown(
            """
**필요한 것**
- OpenAI API Key (리포트 생성용)

**리포트가 안 될 때**
- 로컬: `pip install openai` 설치 확인
- Streamlit Cloud: `requirements.txt`에 `openai` 포함 확인
- 키가 유효하지 않으면(401) 리포트 생성 실패

**참고**
- 이 앱은 ‘티니핑 느낌’의 오리지널 요정 컨셉이며 공식 캐릭터/로고/이미지는 포함하지 않습니다.
"""
        )

st.caption("© AI 습관 트래커 (마법 요정 에디션) — 오늘의 체크가 내일의 마법 ✨")
