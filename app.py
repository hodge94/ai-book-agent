import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from operator import itemgetter
from PyPDF2 import PdfReader
import streamlit.components.v1 as components

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

# 📖 Load QA system
BOOK_PATH = "books/CISA Official Review Manual, 28th Edition[1].pdf"
qa = load_book_qa(BOOK_PATH)

# 💬 Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔈 Text-to-Speech (Browser-safe version)
def browser_speak(text):
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance({repr(text)});
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0)

# 🧠 Display chat history (latest at bottom, like ChatGPT)
for i, (q, a) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# 🔊 Speak checkbox
speak = st.checkbox("🔊 Speak the answer")

# 💬 Input at bottom (ChatGPT style)
with st.container():
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask your question:", key="chat_input", label_visibility="collapsed", placeholder="Type your question here...")
        submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("💡 Thinking..."):
        response = qa.invoke({"question": user_input})
        answer = response.content

        # Save message
        st.session_state.chat_history.append((user_input, answer))

        # Immediately display latest response
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(answer)

        if speak:
            browser_speak(answer)

