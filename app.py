# app.py
import datetime as dt
from typing import Dict, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

# -----------------------------
# Session State
# -----------------------------
def _init_state():
    if "records" not in st.session_state:
        st.session_state["records"] = []  # [{"date": "YYYY-MM-DD", "checked": int, "rate": float, "mood": int}]
    if "ai_report" not in st.session_state:
        st.session_state["ai_report"] = ""
    if "share_text" not in st.session_state:
        st.session_state["share_text"] = ""
    if "last_weather" not in st.session_state:
        st.session_state["last_weather"] = None
    if "last_dog" not in st.session_state:
        st.session_state["last_dog"] = None


_init_state()

# -----------------------------
# API Helpers
# -----------------------------
def get_weather(city: str, api_key: str) -> Optional[Dict]:
    """OpenWeatherMap: 한국어, 섭씨, timeout=10 / 실패 시 None"""
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        return {
            "city": city,
            "temp": j.get("main", {}).get("temp"),
            "feels_like": j.get("main", {}).get("feels_like"),
            "humidity": j.get("main", {}).get("humidity"),
            "desc": (j.get("weather") or [{}])[0].get("description"),
            "icon": (j.get("weather") or [{}])[0].get("icon"),
        }
    except Exception:
        return None


def _breed_from_dog_url(url: str) -> str:
    """Dog CEO URL에서 품종 폴더명을 추출(추정)"""
    try:
        if "/breeds/" not in url:
            return "알 수 없음"
        after = url.split("/breeds/", 1)[1]
        breed_folder = after.split("/", 1)[0]
        return breed_folder.replace("-", " ").strip() or "알 수 없음"
    except Exception:
        return "알 수 없음"


def get_dog_image() -> Optional[Dict]:
    """Dog CEO: 랜덤 강아지 사진 URL+품종(추정), timeout=10 / 실패 시 None"""
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        if j.get("status") != "success":
            return None
        img_url = j.get("message")
        if not img_url:
            return None
        return {"url": img_url, "breed": _breed_from_dog_url(img_url)}
    except Exception:
        return None


def generate_report(
    openai_key: str,
    coach_style: str,
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict],
    dog: Optional[Dict],
) -> Tuple[Optional[str], Optional[str]]:
    """
    OpenAI 리포트 생성 (model: gpt-5-mini)
    ✅ FIX: gpt-5-mini에서 temperature 조절이 불가하므로 temperature 필드를 제거(기본값=1만 사용)
    실패 시 (None, error_message)
    """
    if not openai_key:
        return None, "OpenAI API Key가 없어요. 사이드바에 입력해 주세요."

    style_system = {
        "스파르타 코치": (
            "너는 스파르타 코치다. 짧고 단호하게 말한다. 변명은 컷. "
            "하지만 인신공격은 절대 하지 말고, 구체적인 실행을 강조해라."
        ),
        "따뜻한 멘토": (
            "너는 따뜻한 멘토다. 사용자를 다정하게 격려하되 과장하지 않는다. "
            "실행 가능한 다음 스텝을 부드럽게 제시해라."
        ),
        "게임 마스터": (
            "너는 RPG 게임 마스터다. 오늘을 퀘스트/경험치/레벨업 관점으로 재미있게 묘사한다. "
            "과장된 세계관은 OK지만, 행동은 현실적으로 가능하게 제시해라."
        ),
    }.get(coach_style, "너는 실용적인 습관 코치다. 짧고 명확하게 답해라.")

    habit_lines = [f"- {name}: {'완료' if done else '미완료'}" for name, done in habits.items()]
    habits_text = "\n".join(habit_lines)

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} / {weather.get('desc')} / "
            f"{weather.get('temp')}°C (체감 {weather.get('feels_like')}°C) / 습도 {weather.get('humidity')}%"
        )

    dog_text = "강아지 정보 없음"
    if dog:
        dog_text = f"품종(추정): {dog.get('breed')}"

    system_prompt = (
        f"{style_system}\n\n"
        "출력 형식은 반드시 아래 섹션 제목을 그대로 사용해라(한국어).\n"
        "1) 컨디션 등급: S/A/B/C/D 중 하나\n"
        "2) 습관 분석: (짧게, 핵심 3줄 이내)\n"
        "3) 날씨 코멘트: (날씨에 맞춘 조언 1~2문장)\n"
        "4) 내일 미션: (불릿 3개)\n"
        "5) 오늘의 한마디: (한 문장)\n"
        "추가 규칙: 설교 금지, 과장 금지, 실행 가능한 내용만."
    )

    user_prompt = (
        f"[오늘 습관 체크]\n{habits_text}\n\n"
        f"[기분 점수]\n{mood}/10\n\n"
        f"[날씨]\n{weather_text}\n\n"
        f"[강아지 보상]\n{dog_text}\n"
    )

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-5-mini",
            # ✅ temperature 제거 (gpt-5-mini는 기본값(1)만 지원)
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            try:
                err = r.json()
                msg = err.get("error", {}).get("message", r.text)
            except Exception:
                msg = r.text
            return None, f"OpenAI 호출 오류: {msg}"

        content = r.json()["choices"][0]["message"]["content"]
        return content, None
    except requests.exceptions.Timeout:
        return None, "OpenAI 요청 시간이 초과됐어요. 네트워크 상태를 확인하고 다시 시도해 주세요."
    except requests.exceptions.RequestException:
        return None, "OpenAI 네트워크 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
    except Exception as e:
        return None, f"알 수 없는 오류: {e}"


# -----------------------------
# Demo data (6 days) + today
# -----------------------------
def _seed_demo_records_if_needed():
    if st.session_state["records"]:
        return
    today = dt.date.today()
    sample = [
        (today - dt.timedelta(days=6), 3, 60.0, 6),
        (today - dt.timedelta(days=5), 2, 40.0, 5),
        (today - dt.timedelta(days=4), 4, 80.0, 7),
        (today - dt.timedelta(days=3), 3, 60.0, 6),
        (today - dt.timedelta(days=2), 5, 100.0, 8),
        (today - dt.timedelta(days=1), 1, 20.0, 4),
    ]
    st.session_state["records"] = [
        {"date": d.isoformat(), "checked": checked, "rate": rate, "mood": mood} for d, checked, rate, mood in sample
    ]


def _upsert_today_record(checked_count: int, rate: float, mood: int):
    today_str = dt.date.today().isoformat()
    recs = st.session_state["records"]
    for r in recs:
        if r.get("date") == today_str:
            r["checked"] = checked_count
            r["rate"] = rate
            r["mood"] = mood
            return
    recs.append({"date": today_str, "checked": checked_count, "rate": rate, "mood": mood})


_seed_demo_records_if_needed()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_key = st.text_input("OpenAI API Key", type="password", value="")
    weather_key = st.text_input("OpenWeatherMap API Key", type="password", value="")

    st.divider()
    if st.button("🧹 오늘 리포트/공유텍스트 지우기", use_container_width=True):
        st.session_state["ai_report"] = ""
        st.session_state["share_text"] = ""
        st.success("오늘 출력 결과를 초기화했어요.")


# -----------------------------
# Main UI
# -----------------------------
st.title("AI 습관 트래커")

CITY_CHOICES = [
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
city = st.selectbox("도시 선택", CITY_CHOICES, index=0)

coach_style = st.radio("코치 스타일", ["스파르타 코치", "따뜻한 멘토", "게임 마스터"], horizontal=True)

st.subheader("✅ 습관 체크인")

# 체크박스 5개를 2열 배치 + 이모지
col1, col2 = st.columns(2)
with col1:
    h_wake = st.checkbox("🌅 기상 미션", key="habit_wake")
    h_water = st.checkbox("💧 물 마시기", key="habit_water")
    h_study = st.checkbox("📚 공부/독서", key="habit_study")
with col2:
    h_workout = st.checkbox("🏃 운동하기", key="habit_workout")
    h_sleep = st.checkbox("😴 수면", key="habit_sleep")

mood = st.slider("기분 점수", min_value=1, max_value=10, value=6, step=1)

habits = {
    "기상 미션": bool(h_wake),
    "물 마시기": bool(h_water),
    "공부/독서": bool(h_study),
    "운동하기": bool(h_workout),
    "수면": bool(h_sleep),
}

checked_count = sum(1 for v in habits.values() if v)
rate = round((checked_count / 5) * 100.0, 1)

# 오늘 데이터 자동 저장
_upsert_today_record(checked_count=checked_count, rate=rate, mood=mood)

st.divider()

# Metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("달성률", f"{rate:.1f}%")
with m2:
    st.metric("달성 습관", f"{checked_count}/5")
with m3:
    st.metric("기분", f"{mood}/10")

# -----------------------------
# 7-day bar chart (Altair v6 safe)
# -----------------------------
today = dt.date.today()
window_dates = [today - dt.timedelta(days=i) for i in range(6, -1, -1)]

rec_map = {r["date"]: r for r in st.session_state["records"]}
chart_rows = []
for d in window_dates:
    d_str = d.isoformat()
    r = rec_map.get(d_str)
    chart_rows.append(
        {
            "date": d,
            "달성률": float((r or {}).get("rate", 0.0)),
            "달성개수": int((r or {}).get("checked", 0)),
        }
    )

st.subheader("📈 최근 7일 달성률")

df_chart = pd.DataFrame(chart_rows)
df_chart["date"] = pd.to_datetime(df_chart["date"])

chart = (
    alt.Chart(df_chart)
    .mark_bar()
    .encode(
        x=alt.X("date:T", title="날짜", axis=alt.Axis(format="%m-%d")),
        y=alt.Y("달성률:Q", title="달성률(%)"),
        tooltip=[
            alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
            alt.Tooltip("달성률:Q", title="달성률(%)"),
            alt.Tooltip("달성개수:Q", title="달성개수"),
        ],
    )
    .properties(height=260)
)
st.altair_chart(chart, use_container_width=True)

st.divider()

# -----------------------------
# Result area: Weather + Dog + AI report
# -----------------------------
st.subheader("🧠 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", use_container_width=True)

if btn:
    weather = get_weather(city, weather_key)
    dog = get_dog_image()

    st.session_state["last_weather"] = weather
    st.session_state["last_dog"] = dog

    report, err = generate_report(
        openai_key=openai_key,
        coach_style=coach_style,
        habits=habits,
        mood=mood,
        weather=weather,
        dog=dog,
    )

    if err:
        st.session_state["ai_report"] = ""
        st.session_state["share_text"] = ""
        st.error(err)
    else:
        st.session_state["ai_report"] = report or ""

        share = []
        share.append(f"📅 {dt.date.today().isoformat()} | AI 습관 트래커")
        share.append(f"✅ 달성: {checked_count}/5 ({rate:.1f}%) | 🙂 기분: {mood}/10")
        share.append("— 체크인 —")
        for k, v in habits.items():
            share.append(f"- {k}: {'✅' if v else '⬜'}")
        if st.session_state["last_weather"]:
            w = st.session_state["last_weather"]
            share.append("— 날씨 —")
            share.append(f"- {w.get('city')} / {w.get('desc')} / {w.get('temp')}°C / 습도 {w.get('humidity')}%")
        if st.session_state["last_dog"]:
            d = st.session_state["last_dog"]
            share.append("— 오늘의 보상 강아지 —")
            share.append(f"- {d.get('breed')}")
        share.append("— AI 리포트 —")
        share.append(st.session_state["ai_report"].strip())

        st.session_state["share_text"] = "\n".join(share)

# Weather + Dog cards
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("### 🌦️ 날씨")
    weather = st.session_state.get("last_weather")
    if weather_key and weather:
        st.write(f"**도시:** {weather.get('city')}")
        st.write(f"**상태:** {weather.get('desc')}")
        st.write(f"**기온:** {weather.get('temp')}°C (체감 {weather.get('feels_like')}°C)")
        st.write(f"**습도:** {weather.get('humidity')}%")
    elif weather_key and not weather:
        st.info("날씨를 가져오지 못했어요. (도시명/키/네트워크 확인)")
    else:
        st.info("OpenWeatherMap API Key를 입력하면 날씨를 보여줄게요.")

with c_right:
    st.markdown("### 🐶 오늘의 강아지 보상")
    dog = st.session_state.get("last_dog")
    if dog:
        st.image(dog.get("url"), use_container_width=True)
        st.caption(f"품종(추정): {dog.get('breed')}")
    else:
        st.info("아직 강아지 보상이 없어요. '컨디션 리포트 생성'을 눌러보세요.")

# AI report
st.markdown("### 📄 AI 리포트")
if st.session_state.get("ai_report"):
    st.markdown(st.session_state["ai_report"])
else:
    st.caption("리포트를 생성하면 여기에 표시됩니다.")

# Share text
st.markdown("### 🔗 공유용 텍스트")
if st.session_state.get("share_text"):
    st.code(st.session_state["share_text"], language="text")
else:
    st.caption("리포트를 생성하면 공유용 텍스트가 만들어집니다.")

# API 안내
with st.expander("📌 API 안내 / 사용 방법"):
    st.markdown(
        """
**1) OpenAI API**
- 사이드바에 OpenAI API Key를 입력하면, '컨디션 리포트 생성' 시 AI 코칭 리포트를 생성합니다.
- 모델: `gpt-5-mini`
- 참고: 일부 모델은 `temperature` 값을 지원하지 않아(기본값 1만 허용) 이 앱은 temperature를 보내지 않습니다.

**2) OpenWeatherMap API**
- 사이드바에 OpenWeatherMap API Key를 입력하면, 선택 도시의 현재 날씨를 가져옵니다.
- 한국어(`lang=kr`), 섭씨(`units=metric`)

**3) Dog API (Dog CEO)**
- 별도 키 없이 동작합니다.
- 습관 리포트 생성 시 랜덤 강아지 이미지를 가져와 보상 카드로 보여줍니다.

**네트워크/권한**
- 외부 요청은 Weather/Dog `timeout=10`, OpenAI `timeout=30`입니다.
- 실패하면 None 처리 또는 오류 메시지를 표시합니다.
"""
    )
