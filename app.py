import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from readers import READERS, read_file

load_dotenv()

PROMPT = """ You are a helpful assistant. Answer the question using only the context below and try to give a detailed response.
If the answer is not in the context or if you are unsure, say you could not find it in the document.
If you are not able to answer the question, try to analyze the question and provide a detailed response based on your understanding but don't hallucinate anything randomly and re-asure that answer you are going to give match the question requirements.
Don't make up an answer.
Don't include any information that is not in the context.
Don't hallucinate or fabricate information.

Context:
{context}

Question: {question}"""


@st.cache_resource(show_spinner="Indexing file...")
def build_vectorstore(name, data):
    sections = read_file(name, data)
    if not sections:
        raise ValueError("no readable text found (scanned PDF or image?)")
    docs = [Document(page_content=text, metadata={"location": location}) for location, text in sections]
    chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(docs)
    st.write(f"Total Chunks: {len(chunks)}")
    return Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=uuid.uuid4().hex,
    )


def delete_file(vectorstore):
    vectorstore.delete_collection()
    build_vectorstore.clear()
    st.session_state.uploader_key += 1

def show_sources(sources):
    with st.expander("Sources"):
        for location, text in sources:
            st.markdown(f"**{location}**")
            st.caption(text)

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Set OPENAI_API_KEY in .env, then restart the app.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

vectorstore = None
with st.sidebar:
    file = st.file_uploader(
        "Upload a file",
        type=[ext.lstrip(".") for ext in READERS],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if file:
        try:
            vectorstore = build_vectorstore(file.name, file.getvalue())
        except Exception as error:
            st.error(f"Could not index {file.name}: {error}")
    if vectorstore:
        st.button("🗑️ Delete file", on_click=delete_file, args=(vectorstore,), width="stretch")
    st.button("🧹 Clear conversation", on_click=st.session_state.messages.clear, width="stretch")

if not vectorstore:
    st.info("👈 Upload a file in the sidebar to start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("Ask something about your file...", disabled=not vectorstore):
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                docs = vectorstore.similarity_search(question, k=5)
                context = "\n\n".join(doc.page_content for doc in docs)
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
                answer = llm.invoke(PROMPT.format(context=context, question=question)).content
        except Exception as error:
            st.error(f"Could not get an answer: {error}")
            st.stop()
        sources = [(doc.metadata["location"], doc.page_content) for doc in docs]
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources},
    ]
