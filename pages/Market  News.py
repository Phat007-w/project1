import os
import time
import requests
import streamlit as st
from dateutil.parser import parse

# ------------------ CONFIG ------------------
st.set_page_config(page_title="ข่าวตลาดหุ้น (Market News)", page_icon="🗞", layout="wide")

# ------------------ SECRETS ------------------
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")
if not NEWS_API_KEY:
    st.error("❌ ยังไม่ได้ตั้งค่า NEWS_API_KEY ใน st.secrets.toml หรือ Streamlit Cloud Secrets")
    st.stop()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
use_ai_summary = OPENAI_API_KEY is not None

if use_ai_summary:
    try:
        from openai import OpenAI
        oa_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        use_ai_summary = False

# ------------------ UI HEADER ------------------
st.markdown("## 🗞 ข่าวตลาดหุ้น (Market News)")
st.caption("Thai + Global business headlines via NewsAPI • มีปุ่มสรุปข่าว 3 บรรทัด (ถ้าตั้งค่า OpenAI key)")

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.subheader("Filters")
lang = st.sidebar.selectbox(
    "ภาษา (language)", ["ไทย (th)", "English (en)", "ไทย + อังกฤษ (both)"], index=2
)
query = st.sidebar.text_input("ค้นหาคีย์เวิร์ด (เช่น: SET, Fed, Oil)", value="")
source_scope = st.sidebar.selectbox(
    "แหล่งข่าว", ["Business/Finance headlines", "All news (everything)"], index=0
)
page_size = st.sidebar.slider("จำนวนข่าว/หน้า", 5, 30, 12, step=1)
sort_by = st.sidebar.selectbox("เรียงโดย", ["publishedAt", "relevancy", "popularity"], index=0)
st.sidebar.info("Tip: ถ้าอยากให้โชว์เฉพาะข่าวไทย ให้เลือกภาษา 'ไทย (th)'")

# ------------------ HELPERS ------------------
def build_endpoints():
    """Prepare list of (url, params) to fetch according to user's language choices."""
    endpoints = []
    api_base = "https://newsapi.org/v2"
    headers = {"X-Api-Key": NEWS_API_KEY}

    def _top(lang_code):
        return (f"{api_base}/top-headlines", {"language": lang_code, "category": "business", "pageSize": page_size})

    def _everything(lang_code):
        params = {
            "language": lang_code,
            "q": query or "stock OR economy OR finance OR market",
            "sortBy": sort_by,
            "pageSize": page_size,
        }
        return (f"{api_base}/everything", params)

    if "both" in lang:
        if source_scope.startswith("Business"):
            endpoints.append(_top("th"))
            endpoints.append(_top("en"))
        else:
            endpoints.append(_everything("th"))
            endpoints.append(_everything("en"))
    else:
        code = "th" if "ไทย" in lang else "en"
        if source_scope.startswith("Business"):
            endpoints.append(_top(code))
        else:
            endpoints.append(_everything(code))

    return endpoints, headers


def fetch_news():
    endpoints, headers = build_endpoints()
    all_articles = []
    for url, params in endpoints:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "ok":
                all_articles.extend(data.get("articles", []))
        except Exception as e:
            st.warning(f"โหลดข่าวจาก {url} ไม่สำเร็จ: {e}")
        time.sleep(0.2)  # กัน rate-limit

    # ลบข่าวซ้ำด้วย title + url
    seen = set()
    unique_articles = []
    for a in all_articles:
        key = (a.get("title"), a.get("url"))
        if key not in seen and a.get("title") and a.get("url"):
            seen.add(key)
            unique_articles.append(a)

    # เรียงใหม่ตามเวลาลงข่าว
    unique_articles.sort(
        key=lambda a: parse(a.get("publishedAt", "1970-01-01T00:00:00Z")), reverse=True
    )
    return unique_articles


def summarize_text(text, language_hint="th"):
    """Return 3-bullet summary via OpenAI if available; otherwise None."""
    if not use_ai_summary:
        return None
    try:
        prompt = (
            f"สรุปข่าวนี้เป็น 3 บรรทัดแบบกระชับ ภาษา{'ไทย' if language_hint=='th' else 'อังกฤษ'}:\n\n"
            f"{text}\n\nข้อสรุป:\n"
        )
        resp = oa_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.info(f"สรุปข่าวไม่สำเร็จ: {e}")
        return None


# ------------------ FETCH & RENDER ------------------
articles = fetch_news()
if not articles:
    st.warning("ยังไม่พบข่าวจากเงื่อนไขที่เลือก ลองเปลี่ยนภาษา/คำค้น หรือรอสักครู่")
    st.stop()

# แสดงข่าวแบบ Card
for a in articles:
    title = a.get("title", "—")
    desc = a.get("description") or ""
    url = a.get("url")
    src = (a.get("source") or {}).get("name", "source")
    published = a.get("publishedAt", "")[:16].replace("T", " ")

    st.markdown(f"### [{title}]({url})")
    st.caption(f"Source: {src} • Published: {published}")
    if desc:
        st.write(desc)

    # ใช้ expander สำหรับ summary
    if use_ai_summary:
        summary_key = f"summ_{hash(url)}"
        with st.expander("🧠 สรุปข่าว 3 บรรทัด", expanded=False):
            if st.button("สรุปข่าว", key=summary_key):
                with st.spinner("กำลังสรุปข่าว..."):
                    summary = summarize_text(f"{title}\n{desc or ''}\n{url}",
                                             language_hint=("th" if "ไทย" in lang else "en"))
                    if summary:
                        st.success(summary)

    st.markdown(f"[อ่านข่าวฉบับเต็ม ▶]({url})")
    st.markdown("---")
