import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- Constants ---
PDF_PATH = "data/finance_bill_2025.pdf"
CHROMA_DIR = "finance_bill_vectorstore"
MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"


# --- Embedding Setup ---
bge_model = SentenceTransformer("BAAI/bge-base-en")

class BGEEmbeddings:
    def embed_documents(self, texts):
        return bge_model.encode(texts, batch_size=8, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return bge_model.encode([text], normalize_embeddings=True).tolist()[0]

embedder = BGEEmbeddings()

# --- Streamlit UI ---
st.set_page_config(page_title="Finance Bill Chatbot",
                    page_icon="/home/jones/rag_project/image_icons/title.png")
st.title("Chat with the Finance Bill 2025")
st.markdown("Ask any question about the Finance Bill 2025")

# --- Force Rebuild Vectorstore if missing ---
if not os.path.exists(os.path.join(CHROMA_DIR, "chroma.sqlite3")):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

# --- Load and Process PDF ---
@st.cache_resource
def build_retriever():
    loader = PyPDFLoader(PDF_PATH)  # update this if using a different path
    docs = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=700)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=700)

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

    # Step 1: Split into parent chunks
    parent_docs = parent_splitter.split_documents(docs)

    # Step 2: Split into child chunks and attach parent IDs
    child_docs = []
    for i, parent_doc in enumerate(parent_docs):
        parent_id = str(i)
        children = child_splitter.split_documents([parent_doc])
        for child in children:
            child.metadata["parent_id"] = parent_id
        child_docs.extend(children)

    # Step 3: Embed child docs and create vectorstore
    vectorstore = FAISS.from_documents(child_docs, embedder)

    # Step 4: Create document store with parent docs
    store = InMemoryStore()
    store.mset([(str(i), doc) for i, doc in enumerate(parent_docs)])

    # Step 5: Build ParentDocumentRetriever manually
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever

retriever = build_retriever()

# --- LLM Setup ---
llm = ChatGroq(
    model_name=MODEL_NAME,
    api_key=api_key
)

# --- Prompt for final answer generation ---
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that only answers questions using information from the Finance Bill 2025.
When prompted with greetings, you can start the conversation in a welcoming tone
 e.g Hi, I'm hear to help you understand more on the finance bill 2025
If the question is not related to this bill, say:
 "I can only answer questions about the Kenyan Finance Bill 2025."

Context: {context}
Question: {question}
Answer:"""
)

# --- Prompt for question reformulation ---
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow-Up Question:
{question}

Standalone question:"""
)

# --- Memory ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  
)


# --- Subchains ---
combine_docs_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=answer_prompt)
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

# --- Final Chain ---
qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain,
    question_generator=question_generator,
    memory=memory,
    return_source_documents=True,
    output_key="answer"  # 👈 this tells memory which field to store
)


# --- Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something about the Finance Bill 2025...")

if user_input:
    with st.spinner("Thinking..."):
        result = qa_chain({"question": user_input})
        answer = result["answer"]
        sources = result["source_documents"]

        st.session_state.chat_history.append((user_input, answer))

        # Optional: Show retrieved sources for debugging
        with st.expander("🇰🇪 Retrieved Sources", expanded=False):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}**")
                st.text(doc.page_content[:500])  # First 500 characters

# --- Display Chat History ---
for q, a in st.session_state.chat_history:
    st.chat_message("user").markdown(q)
    st.chat_message("assistant").markdown(a)

