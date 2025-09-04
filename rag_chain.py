import os
import traceback
import docx
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Global variables
vectorstore = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def extract_text(file_path):
    """
    Extract text from TXT, PDF, DOCX, MD files.
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".pdf":
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text

        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])

        else:
            return "⚠️ Unsupported file format."

    except Exception as e:
        traceback.print_exc()
        return f"⚠️ Error extracting text: {str(e)}"


def ingest_file(file_path):
    """
    Process uploaded file into embeddings and store in vector DB.
    """
    global vectorstore

    text = extract_text(file_path)
    if not text.strip():
        raise ValueError("No text extracted from file.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_texts([text], embedding=embeddings, persist_directory="./chroma_store")
    vectorstore.persist()
    return "File ingested successfully."


def get_conversation_chain():
    """
    Build a retrieval-augmented generation chain with memory.
    """
    global vectorstore
    llm = OllamaLLM(model="phi3")

    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""You are a helpful assistant. Use chat history + context if available.
Chat History:
{chat_history}

Question: {question}
Answer:""",
    )

    if vectorstore:
        retriever = vectorstore.as_retriever()
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
        return chain
    else:
        return LLMChain(llm=llm, prompt=prompt, memory=memory)


def invoke(query: str) -> str:
    try:
        chain = get_conversation_chain()
        return chain.run({"question": query})
    except Exception as e:
        traceback.print_exc()
        return f"⚠️ Error generating response: {str(e)}"


def is_ready() -> bool:
    """
    Simple readiness check for the RAG pipeline.
    Returns True if Ollama is available and chain can be initialized.
    """
    try:
        if vectorstore:
            _ = get_conversation_chain()
        else:
            _ = OllamaLLM(model="phi3")  # fallback check
        return True
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")
        return False
