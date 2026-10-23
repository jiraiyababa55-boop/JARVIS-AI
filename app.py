import hashlib
import io
import os
import re
from datetime import date

import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image
from pypdf import PdfReader

# ------------------------------------------------------------------
# SETTINGS  (edit these to customize your assistant)
# ------------------------------------------------------------------
MODEL = "gemini-2.5-flash"  # check aistudio.google.com for the newest free model name
MAX_MATERIAL_CHARS = 40000  # how much of your PDF the AI reads

BASE_PROMPT = """You are Jarvis, a friendly personal AI assistant.
Explain things in simple words, step by step, with short sentences.
Be honest. If you are not sure, say so. Never invent facts or numbers."""

MODES = {
    "Study tutor": {
        "search": False,
        "prompt": """You are a patient tutor for any subject.
If a screenshot is given, read the question carefully, solve it step by step,
and explain WHY each step is done. End with a one-line summary.
If study material is provided, answer from it first and say which part you used.""",
    },
    "News & World": {
        "search": True,
        "prompt": """You are a news and world-events assistant.
Use Google Search to get today's information. Give the 5 most important stories
as short, simple bullet points: what happened, where, and why it matters.
Also answer weather questions using live search. Mention the source names.""",
    },
    "Markets (info only)": {
        "search": True,
        "prompt": """You are a markets education assistant.
Use Google Search to get current market information: indexes, top gainers and
losers, and news behind the moves. Explain terms in simple words.
NEVER say 'buy this' or 'sell this' and never promise profit. Instead, show data,
list pros and risks, and explain how a beginner could research further.
Always end with: 'This is information, not financial advice. Markets are risky.'""",
    },
    "Free chat": {
        "search": True,
        "prompt": "Answer any question helpfully. Use Google Search when the answer needs current information.",
    },
}

LEVELS = [
    "Primary school (grades 1-5)",
    "Middle school (grades 6-8)",
    "High school (grades 9-12)",
    "College / University",
    "Adult / Professional",
]

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Urdu": "ur",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "Bengali": "bn",
}

# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Jarvis AI Assistant", page_icon="🎙️")
st.title("🎙️ Jarvis AI Assistant")


def get_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


API_KEY = get_key()
if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it in Streamlit 'Secrets' (see the guide).")
    st.stop()

client = genai.Client(api_key=API_KEY)

if "history" not in st.session_state:
    st.session_state.history = []   # what the AI remembers
    st.session_state.messages = []  # what is shown on screen
    st.session_state.last_audio = None

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", list(MODES.keys()))
level = st.sidebar.selectbox("My level", LEVELS, index=2)
language = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
speak = st.sidebar.toggle("Speak answers aloud", value=True)

st.sidebar.divider()
pdf_file = st.sidebar.file_uploader("Study material (PDF)", type=["pdf"])


@st.cache_data(show_spinner=False)
def read_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return text[:MAX_MATERIAL_CHARS]


material = ""
if pdf_file:
    material = read_pdf(pdf_file.getvalue())
    st.sidebar.success(f"Loaded {len(material):,} characters")

if st.sidebar.button("New chat"):
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.last_audio = None
    st.rerun()

if mode == "Markets (info only)":
    st.warning("Information only, not financial advice. Markets are risky.")


def build_system_prompt() -> str:
    parts = [
        BASE_PROMPT,
        MODES[mode]["prompt"],
        f"Today's date is {date.today().strftime('%A, %d %B %Y')}.",
        f"The user's level is: {level}. Explain at that level.",
        f"Reply in {language}.",
    ]
    if material:
        parts.append("STUDY MATERIAL provided by the user:\n" + material)
    return "\n\n".join(parts)


def clean_for_speech(text: str) -> str:
    return re.sub(r"[*#_`>|~]", "", text)[:1500]


# ------------------------------------------------------------------
# CHAT LOGIC
# ------------------------------------------------------------------
def ask(current_parts, label):
    st.session_state.messages.append(("user", label))
    with st.chat_message("user"):
        st.write(label)

    contents = st.session_state.history + [
        types.Content(role="user", parts=current_parts)
    ]
    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(),
        tools=[types.Tool(google_search=types.GoogleSearch())]
        if MODES[mode]["search"]
        else None,
    )

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = client.models.generate_content(
                    model=MODEL, contents=contents, config=config
                )
            reply = response.text or "I could not make an answer. Please try again."
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return
        st.write(reply)

        if speak:
            try:
                mp3 = io.BytesIO()
                gTTS(clean_for_speech(reply), lang=LANGUAGES[language]).write_to_fp(mp3)
                st.audio(mp3.getvalue(), format="audio/mp3", autoplay=True)
            except Exception:
                st.caption("(Voice playback failed, text answer is above.)")

    # Save short text-only memory (not images/audio) to keep it fast
    st.session_state.history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=label)])
    )
    st.session_state.history.append(
        types.Content(role="model", parts=[types.Part.from_text(text=reply)])
    )
    st.session_state.messages.append(("assistant", reply))


# ------------------------------------------------------------------
# SCREEN
# ------------------------------------------------------------------
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

uploaded = st.file_uploader(
    "Screenshot or photo of a question (optional)", type=["png", "jpg", "jpeg", "webp"]
)
if uploaded:
    st.image(uploaded, width=250)

audio = st.audio_input("🎤 Tap to speak")
typed = st.chat_input("Or type your question...")

image_parts = []
if uploaded:
    img = Image.open(uploaded)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    image_parts = [types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")]

if audio:
    data = audio.getvalue()
    key = hashlib.md5(data).hexdigest()
    if key != st.session_state.last_audio:  # handle each recording only once
        st.session_state.last_audio = key
        voice = types.Part.from_bytes(data=data, mime_type="audio/wav")
        ask(
            image_parts + [voice, types.Part.from_text(text="Answer my spoken question.")],
            "🎤 (voice question)",
        )
elif typed:
    ask(image_parts + [types.Part.from_text(text=typed)], typed)
elif uploaded and st.button("Solve this screenshot"):
    ask(
        image_parts + [types.Part.from_text(text="Solve this step by step and explain.")],
        "📷 (screenshot question)",
    )
