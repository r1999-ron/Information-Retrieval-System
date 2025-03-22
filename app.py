# 1. Imports and Configuration
import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, Date, Enum
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker,declarative_base
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
from sqlalchemy.future import select
from sqlalchemy import LargeBinary
from sqlalchemy.exc import IntegrityError
from langchain.schema import Document
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta, date
import bcrypt
import jwt
import logging
import time
import pickle

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

VALID_TOKENS = {
    "abcdef": "upload",
    "qwerty": "query"
}

# 2. Database Models
Base = declarative_base()

class FAISSIndex(Base):
    __tablename__ = "faiss_index"
    id = Column(Integer, primary_key=True, index=True)
    index_data = Column(LargeBinary)  # Store the serialized FAISS index
    created_at = Column(Date, default=datetime.utcnow)  # Track when the index was created


# 3. Database Setup
DATABASE_URL = "sqlite+aiosqlite:///./employee.db"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except HTTPException:
            # Re-raise HTTPExceptions (e.g., 404) without modification
            raise
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise HTTPException(status_code=500, detail="Database error")
        finally:
            await session.close()

# 4. FastAPI App and Event Handlers
@app.on_event("startup")
async def startup_event():
    # Manually create a database session
    async with AsyncSessionLocal() as db:
        await init_db()
        await load_faiss_index(db)

# 5. Utility Function
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE_MB = 5  # Limit file size to 5MB

# PDF Processing Functions
async def save_file(file: UploadFile) -> str:
    """Save uploaded file asynchronously in chunks."""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    return file_path

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from a file (PDF or TXT)."""
    if file_path.endswith(".pdf"):
        text = ""
        try:
            pdf_reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")
        return text
    elif file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading TXT file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and TXT files are allowed.")

def split_text_into_chunks(text: str) -> List[str]:
    """Splits text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

async def process_file(file: UploadFile) -> List[str]:
    """Handles file saving, text extraction, and chunking."""
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit.")

    file_path = await save_file(file)
    text = ""
    try:
        text = extract_text_from_file(file_path)
    except HTTPException as e:
        raise e
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    return split_text_into_chunks(text)

class APIRateLimitError(Exception):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry=retry_if_exception_type(APIRateLimitError))
def get_llm_with_retry():
    try:
        return ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "quota exceeded" in str(e).lower():
            raise APIRateLimitError(f"Rate limit error: {e}")
        raise

def get_conversational_chain(with_memory: bool = True):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No FAISS index loaded. Please upload documents first.")
    # Log the size of the FAISS index
    logger.info(f"Using FAISS index with {vector_store.index.ntotal} embeddings.")
    llm = get_llm_with_retry()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if with_memory else None
    # Use the entire FAISS index for retrieval
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# 6. FAISS Index Management
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

async def save_faiss_index_to_db(all_text_chunks: List[str], db: AsyncSession):
    """Save the FAISS index to the database."""
    global vector_store

    try:
        # Create documents from the new text chunks
        documents = [Document(page_content=text) for text in all_text_chunks]

        # Log the number of new documents being added
        logger.info(f"Adding {len(documents)} new documents to the FAISS index.")

        # If no existing vector store, create a new one
        if vector_store is None:
            logger.info("No existing FAISS index found. Creating a new one.")
            vector_store = FAISS.from_documents(documents, embeddings_model)
        else:
            # Log the size of the existing index
            logger.info(f"Merging new embeddings with existing FAISS index (size: {vector_store.index.ntotal}).")
            # Merge new embeddings with the existing vector store
            new_vector_store = FAISS.from_documents(documents, embeddings_model)
            vector_store.merge_from(new_vector_store)
            # Log the size of the updated index
            logger.info(f"Updated FAISS index size: {vector_store.index.ntotal}.")

        # Serialize the FAISS index
        serialized_index = pickle.dumps(vector_store)

        # Save the serialized index to the database
        db_index = FAISSIndex(index_data=serialized_index)
        db.add(db_index)
        await db.commit()
        logger.info("FAISS index saved to the database.")
    except Exception as e:
        logger.error(f"Error saving FAISS index to the database: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save FAISS index.")

async def load_faiss_index_from_db(db: AsyncSession):
    """Load the FAISS index from the database."""
    global vector_store

    logger.info("Loading FAISS index from the database.")

    try:
        # Fetch the latest FAISS index from the database
        result = await db.execute(select(FAISSIndex).order_by(FAISSIndex.created_at.desc()))
        db_index = result.scalars().first()

        if db_index:
            logger.info("FAISS index found in the database.")
            try:
                # Deserialize the FAISS index
                vector_store = pickle.loads(db_index.index_data)
                logger.info(f"FAISS index loaded from the database (size: {vector_store.index.ntotal}).")
            except Exception as e:
                logger.error(f"Failed to deserialize FAISS index: {e}")
                vector_store = None
        else:
            logger.warning("No FAISS index found in the database, starting fresh.")
            vector_store = None
    except Exception as e:
        logger.error(f"Error loading FAISS index from the database: {e}")
        vector_store = None

async def save_faiss_index(all_text_chunks: List[str], db: AsyncSession):
    """Save the FAISS index to the database."""
    await save_faiss_index_to_db(all_text_chunks, db)

async def load_faiss_index(db: AsyncSession):
    """Load the FAISS index from the database."""
    await load_faiss_index_from_db(db)

# 7. API Endpoints
@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    token: str = Header(None),
    db: AsyncSession = Depends(get_db)  # Inject the database session
):
    start_time = time.time()

    if token not in VALID_TOKENS or VALID_TOKENS[token] != "upload":
        logger.error(f"Unauthorized access attempt with token: {token}")
        raise HTTPException(status_code=404, detail="Unauthorized")

    global vector_store
    all_text_chunks = []
    file_processing_times = []

    for file in files:
        file_start_time = time.time()
        try:
            # Log file size
            file_size_mb = file.size / (1024 * 1024)
            logger.info(f"Processing file: {file.filename} (Size: {file_size_mb:.2f} MB)")

            # Process the file
            chunks = await process_file(file)
            all_text_chunks.extend(chunks)

            # Log processing time for this file
            file_end_time = time.time()
            file_processing_time = file_end_time - file_start_time
            file_processing_times.append(file_processing_time)
            logger.info(f"File {file.filename} processed in {file_processing_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")

    # Log total file processing time
    total_file_processing_time = time.time() - start_time
    logger.info(f"Total file processing time: {total_file_processing_time:.2f} seconds.")

    # Save the updated FAISS index
    faiss_start_time = time.time()
    await save_faiss_index(all_text_chunks, db)  # Pass the db session
    faiss_end_time = time.time()
    faiss_indexing_time = faiss_end_time - faiss_start_time
    logger.info(f"FAISS indexing completed in {faiss_indexing_time:.2f} seconds.")

    # Log total upload and processing time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Upload and processing completed in {total_time:.2f} seconds.")

    return {"message": f"{len(files)} files uploaded and processed successfully."}

# Query Models
class QueryRequest(BaseModel):
    question: str
    chat_history: List[str] = []

def extract_answer(response):
    return response.get("answer", "No answer found.")

@app.post("/query")
async def query_document(data: QueryRequest, token: str = Header(None)):
    if token not in VALID_TOKENS or VALID_TOKENS[token] not in ["upload", "query"]:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token")
    # Clear memory before processing the query
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = get_conversational_chain(with_memory=True)
    
    response = chain.invoke(vars(data))
    return {"answer": extract_answer(response)}

@app.get("/faiss-index/")
async def inspect_faiss_index(db: AsyncSession = Depends(get_db)):
    """Inspect the FAISS index metadata."""
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=404, detail="No FAISS index loaded.")

    return {
        "index_size": vector_store.index.ntotal,
        "index_dimensions": vector_store.index.d,
    }

@app.delete("/clear-faiss-index/")
async def clear_faiss_index(db: AsyncSession = Depends(get_db)):
    """Clear all rows from the faiss_index table."""
    await db.execute(FAISSIndex.__table__.delete())
    await db.commit()
    return {"message": "FAISS index cleared successfully."}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application."}

@app.get("/health/")
async def health_check():
    return {"status": "ok"}