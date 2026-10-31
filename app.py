
import streamlit as st
from google import genai
from pypdf import PdfReader
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CivicSaarthi AI",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
}
.hero {
    background: linear-gradient(120deg,#102f56,#2475a9);
    padding: 30px;
    border-radius: 16px;
    color: white;
}
.hero h1 {color:white;}
.hero p {color:#eaf3ff;}
.stMarkdown {
    color: #222222 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏛️ CivicSaarthi AI</h1>
<p>Your Intelligent Digital Governance Assistant</p>
<p>Making Government Information Simple and Accessible</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    language = st.sidebar.selectbox(
    "🌐 Select Response Language",
    ["Auto-detect", "English", "Hindi", "Hinglish"]
    )

    st.divider()
    st.header("📄 Upload Documents")

    files = st.file_uploader(
        "Upload government PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.caption("Upload official government notices and scheme guidelines.")

@st.cache_data(show_spinner=False)
def read_pdfs(file_data):
    documents = []

    for name, content in file_data:
        try:
            import io
            reader = PdfReader(io.BytesIO(content))

            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append({
                        "text": text,
                        "source": name,
                        "page": number
                    })
        except Exception:
            continue

    return documents

def retrieve(question, documents):
    if not documents:
        return []

    texts = [d["text"] for d in documents]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)
    query = vectorizer.transform([question])

    scores = cosine_similarity(query, matrix).flatten()

    indices = scores.argsort()[::-1][:8]

    return [
        documents[i]
        for i in indices
        if scores[i] > 0.05
    ]

def ask_ai(question, documents, api_key, history=None, language="Auto-detect"):

    sources = retrieve(question, documents)
    conversation_history = ""

    if history:
        for message in history[-6:]:
            conversation_history += (
                f"{message['role']}: {message['content']}\n"
            )

    if sources:
        context = "\n".join(
            f"Source: {d['source']}, Page: {d['page']}\n{d['text']}"
            for d in sources
        )

        prompt = f"""
You are CivicSaarthi AI, a responsible public information assistant.

Answer using the supplied document extracts.
Use simple language and reply in the selected language: {language}.
If language is Auto-detect, reply in the same language as the user's question.
Never invent eligibility, deadlines or government rules.
If information is missing, clearly say so.
Cite the document name and page.
IMPORTANT ANSWERING RULES:

1. Separate every scheme component clearly.
2. Distinguish student eligibility from institution eligibility.
3. Never apply one component's eligibility rules to another.
4. Mention the source page for every important rule.
5. If information is missing, say "Not specified in the uploaded document."
6. Never assume that rules from an older year are currently valid.
7. Mention the document's year whenever available.
8. Do not present extracted information as verified current government policy.
CONVERSATION HISTORY:
{conversation_history}

Use previous conversation only when it helps understand the current question.
Always prioritize the uploaded document for scheme-related facts.
DOCUMENTS:
{context}

QUESTION:
{question}
"""
    else:
        prompt = f"""
You are CivicSaarthi AI, a responsible government information assistant.

Question: {question}

Reply in the selected language: {language}.
If language is Auto-detect, reply in the same language as the user's question.
Do not invent current government rules, scheme deadlines
or eligibility. Explain that no matching uploaded document
was found and recommend checking official government sources.
"""

    
    client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else api_key
)

    models_to_try = [
        "gemini-3.5-flash",
    ]

    answer = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                answer = response.text or "No response received."
                break

            except Exception as e:
                error_text = str(e)
                st.error(f"Gemini API Error: {error_text}")

                if "503" in error_text or "UNAVAILABLE" in error_text:
                    if attempt == 0:
                        time.sleep(3)
                        continue
                    break

                return "AI service error. Please try again."

        if answer:
            break

    if not answer:
        return "Gemini is busy right now. Please try again in a minute."

    if sources:
        answer += "\n\n### 📚 Retrieved Sources\n"
        for d in sources:
            answer += f"\n- {d['source']} — Page {d['page']}"

    return answer




st.divider()

a, b, c = st.columns(3)

a.metric("Platform", "AI Powered")
b.metric("Documents", len(files) if files else 0)
c.metric("Language", "Multilingual")

st.subheader("💬 Ask CivicSaarthi")

st.write(
    "Ask about government schemes, public notices "
    "and citizen services."
)
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask your question here...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if not api_key:
            answer = "Please enter your Gemini API key in the sidebar."

        else:
            uploaded_data = [
                (f.name, f.getvalue())
                for f in files
            ] if files else []

            documents = read_pdfs(uploaded_data)

            with st.spinner("CivicSaarthi is thinking..."):
                st.info("Question submitted. AI function is running...")
                try:
                    answer = ask_ai(
                        question,
                        documents,
                        api_key,
                        history=st.session_state.messages[:-1],
                        language=language
                    )
                except Exception as e:
                    answer = f"Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

st.divider()

st.caption(
    "CivicSaarthi AI | Digital Governance for Everyone 🇮🇳"
)
