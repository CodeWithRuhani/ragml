import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Initial Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel("gemini-1.5-flash")
print("Model loaded successfully. 🎊🥳")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embeddings model loaded successfully. 🎊🥳")

# --- Vector Store Setup ---
DB_FAISS_PATH = "ml.faiss_index"
vectorstore = None

try:
    doc = TextLoader("mlbookcontent.txt", encoding="utf-8")
    document = doc.load()
    if not document:
        raise ValueError("No document found in the text file.")
    else:
        # Use only the RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(document)

        if os.path.exists(DB_FAISS_PATH):
            vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded from local path. 🥳")
        else:
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(DB_FAISS_PATH)
            print("Vector store created and saved. 🎉🎊")
except Exception as e:
    st.error(f"Error during document processing: {e}")
    raise RuntimeError(f"Failed to initialize vectorstore: {e}")

# __________Backend Logic (Corrected)_________
def generate_response(query):
    """
    Generates a response based on the query, using RAG if relevant keywords are found.
    """
    keywords = [
        "Machine learning", "Artificial Intelligence", "Deep Learning", "Neural Networks",
        "Computer Vision", "Natural Language Processing", "Reinforcement Learning",
        "Supervised Learning", "Unsupervised Learning", "Semi-Supervised Learning",
        "Generative Models", "ML", "AI", "NLP", "CV"
    ]
    
    query_is_relevant = any(keyword.lower() in query.lower() for keyword in keywords)
    prompt = ""
    
    if query_is_relevant and vectorstore:
        relevant_docs = vectorstore.similarity_search(query, k=3)
        if relevant_docs:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"""
            You are an expert in the field of Machine Learning and Artificial Intelligence.
            Use ONLY the following context to answer the user's question.
            Do not use any external knowledge. If the answer is not in the context,
            politely state that you cannot find the information in the provided document.

            Context: {context}

            Question: {query}
            """
        else:
            query_is_relevant = False

    if not query_is_relevant:
        prompt = f"""
        You are an expert in the field of Machine Learning and Artificial Intelligence.
        Answer the question based on your general knowledge.

        Question: {query}
        """
        
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, an error occurred while generating the response: {e}"

# __________Frontend Setup_________

# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Dark Theme CSS
st.markdown("""
    <style>
    .stApp {
        background: #0f1117;
        color: #e4e6eb;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #1c1f2e;
        color: white;
        padding: 20px;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #bb86fc;
    }
    .sidebar-sub {
        font-size: 14px;
        color: #bbb;
    }
    .dev-info {
        margin-top: 20px;
        font-size: 15px;
        color: #ffcc00;
    }
    .history-item {
        background: #2b2f44;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 14px;
        color: white;
        cursor: pointer;
        border: 1px solid #444;
    }
    .history-item:hover {
        background: #3c4260;
    }
    .user-msg {
        background-color: #6a1b9a; /* purple bubble */
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0px;
        font-size: 16px;
        max-width: 75%;
        float: right;
        clear: both;
        color: #fff;
    }
    .bot-msg {
        background-color: #1f2233;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0px;
        font-size: 16px;
        max-width: 75%;
        float: left;
        clear: both;
        color: #e4e6eb;
        border: 1px solid #333;
    }
    .stChatInput input {
        background: #2b2f44 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px !important;
        border: 1px solid #555 !important;
    }
    .typing-indicator {
        font-size: 14px;
        color: #bbb;
        font-style: italic;
        margin: 6px 0;
    }
    .stButton > button {
        background: #9c27b0; /* Darker purple */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 18px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: #7b1fa2;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🤖 RAG Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>PDF-powered assistant for ML & AI.</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='dev-info'>👩‍💻 Developer: Ruhani Gera</div>", unsafe_allow_html=True)
    st.markdown("<div class='dev-info'>💼 Role: Chatbot Developer</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    linkedin_url = "https://www.linkedin.com/in/ruhani-gera-851454300/"
    st.markdown(f"<a href='{linkedin_url}' target='_blank'><button style='background:#bb86fc;color:white;padding:10px 15px;border:none;border-radius:8px;cursor:pointer;width:100%;'>📩 Contact Me</button></a>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<b style='color:#bb86fc;'>✨ Features:</b><br>- RAG-powered responses<br>- Context-aware conversations<br>- Modern UI", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# -------------------- MAIN CHAT INTERFACE --------------------
st.markdown("<h1 style='text-align:center; color:#bb86fc;'>💬 RAG Chatbot on ML/AI</h1>", unsafe_allow_html=True)

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<div class='user-msg'>{text} 🙋</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {text}</div>", unsafe_allow_html=True)

# Chat input field
if user_input := st.chat_input("Ask me anything about Machine Learning..."):
    st.session_state.history.append(("You", user_input))
    with st.chat_message("user"):
        st.markdown(f"<div class='user-msg'>{user_input} 🙋</div>", unsafe_allow_html=True)

    with st.spinner("🤖 Bot is thinking..."):
        answer = generate_response(user_input)
        st.session_state.history.append(("Bot", answer))
    
    st.rerun()