from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import torch

def start_rag_chat(document_text, model_name):
    documents = [Document(page_content=document_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device}
    )

    db = FAISS.from_documents(texts, embeddings)

    llm = OllamaLLM(model=model_name)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    return qa_chain
