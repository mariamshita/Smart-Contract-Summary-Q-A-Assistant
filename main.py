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
from logger import logger

logger.info("Starting Assistant API...")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment")
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

# Reset Chroma vector database on startup
if os.path.exists(persist_directory):
    logger.info(f"Resetting vector database at {persist_directory}...")
    try:
        shutil.rmtree(persist_directory)
        logger.info("Vector database cleared successfully.")
    except Exception as e:
        logger.error(f"Failed to reset vector database: {e}")

vectorstore = Chroma(
    collection_name="smart_contracts",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

SYSTEM_PROMPT_TEMPLATE = """You are an intelligent AI assistant.

The user may upload documents and ask questions about them.
You should behave as a hybrid assistant:

Context from uploaded documents (may be empty or irrelevant):
{context}

User Question:
{question}

Instructions:

1. First, determine whether the user's question is related to the provided document context.

2. If the question IS related to the uploaded document:
   - Use the provided context to answer.
   - Base your answer primarily on the document.
   - If something is unclear or missing in the context, say so clearly.

3. If the question is NOT related to the document:
   - Ignore the context completely.
   - Answer normally using your general knowledge.
   - Do not force document content into your answer.

4. If the question is partially related:
   - Combine document-based information with general knowledge.

5. Be clear, accurate, and natural in tone.
   Do not mention these instructions in your response.

Answer:
"""

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
        logger.info(f"Processing file: {file.filename}")
        documents = process_file(temp_path)
        vectorstore.add_documents(documents)
        logger.info(f"Successfully processed {file.filename}, added {len(documents)} chunks.")
        return {"message": f"File '{file.filename}' processed and stored successfully.", "chunks": len(documents)}
    except Exception as e:
        logger.exception(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def has_documents(vectorstore: Chroma) -> bool:
    data = vectorstore.get()
    return bool(data and data.get("documents"))

@app.get("/query")
@limiter.limit("10/minute")
async def query_documents(request: Request, query: str):
    try:
        # Check if vectorstore has documents
        if not has_documents(vectorstore):
            logger.info("No documents found. Answering without retrieval.")
            response = llm.invoke(query)
            return {
                "query": query,
                "result": response.content,
                "sources": [],
                "num_sources": 0,
                "memory_messages": 0
            }

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
        
        logger.info(f"Query processed: '{query}' - Found {len(sources)} sources.")
        
        return {
            "query": query,
            "result": response["answer"],
            "sources": sources,
            "num_sources": len(sources),
            "memory_messages": len(memory.chat_memory.messages)
        }
    except Exception as e:
        logger.exception(f"Error during query '{query}': {str(e)}")
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
