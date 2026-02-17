import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
import langchain
from langchain_community.cache import SQLiteCache
from langchain_core.prompts import PromptTemplate
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from utils import process_file
from retrieval import get_local_retriever
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Smart Contract Summary & Q&A Assistant",
    description="A RAG system with Local Hybrid Search and Cross-Encoder Reranking."
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
persist_directory = "./chroma_db"

vectorstore = Chroma(
    collection_name="smart_contracts",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

SYSTEM_PROMPT_TEMPLATE = """You are an expert smart contract analyst. Your role is to analyze smart contracts and answer questions accurately based on the provided context.

Context from the smart contract:
{context}

Question: {question}

Instructions:
- Provide accurate, detailed answers based ONLY on the context provided
- If the context doesn't contain enough information, clearly state that
- Highlight any security concerns or important details
- Use technical terminology appropriately
- Be concise but thorough

Answer:"""

qa_prompt = PromptTemplate(
    template=SYSTEM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        documents = process_file(temp_path)
        vectorstore.add_documents(documents)
        return {"message": f"File '{file.filename}' processed and stored successfully.", "chunks": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/query")
@limiter.limit("10/minute")
async def query_documents(request: Request, query: str):
    try:
        retriever = get_local_retriever(vectorstore)
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        response = qa_chain.invoke({"question": query})
        
        sources = []
        for i, doc in enumerate(response.get("source_documents", []), 1):
            source_info = {
                "chunk_id": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "query": query,
            "result": response["answer"],
            "sources": sources,
            "num_sources": len(sources),
            "memory_messages": len(memory.chat_memory.messages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

global_retriever = get_local_retriever(vectorstore)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": global_retriever | format_docs, "question": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)

add_routes(
    app,
    rag_chain,
    path="/rag",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
