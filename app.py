import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from operator import itemgetter
import pyttsx3
from PyPDF2 import PdfReader

# 🔐 Load API key from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 🔐 Token-Protected access
SECRET_TOKEN = "yalistudy2025"
token = st.query_params.get("token", "").strip()

if token != SECRET_TOKEN:
    st.error("❌ Access Denied. You need a valid link to access this app.")
    st.stop()

# 👋 Welcome message
st.markdown("""
### 👋 Hello Yalitza  
I am an AI Agent created by **Marcos Hodge** to assist you with your studies.
""")

# 📚 Load and index book on startup
@st.cache_resource
def load_book_qa(book_path):
    reader = PdfReader(book_path)
    text = "".join([p.extract_text() or "" for p in reader.pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template(
        "Use the context below to answer the question:\n\n{context}\n\nQuestion: {question}"
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

    chain = RunnableMap({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }) | prompt | llm

    return chain

# 📖 Set book path
BOOK_PATH = "books/CISA Official Review Manual, 28th Edition[1].pdf"
qa = load_book_qa(BOOK_PATH)

# 🗣 Text-to-speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 💬 Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 💬 UI
question = st.text_input("💬 Ask me anything about the book:")
speak = st.checkbox("🔊 Speak the answer")

if question:
    with st.spinner("💡 Thinking..."):
        response = qa.invoke({"question": question})
        st.session_state.chat_history.append((question, response.content))
        st.markdown(f"**Answer:** {response.content}")
        if speak:
            speak_text(response.content)

# 📜 Show history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧠 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")

