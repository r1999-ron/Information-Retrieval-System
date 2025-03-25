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
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import event
from sqlalchemy import DateTime  # Add this with your other imports
from typing import Optional
import hashlib
from fastapi import Query


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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

VALID_TOKENS = {
    "abcdef": "upload",
    "qwerty": "query"
}

# Dictionary to store separate vector stores for each document type
vector_stores = {
    'A': None,
    'B': None,
    'C': None
}

# Access control rules - defines which document types each employee type can access
ACCESS_RULES = {
    'A': ['A'],
    'B': ['A', 'B'],
    'C': ['A', 'B', 'C']
}

# 2. Database Models
Base = declarative_base()

class FAISSIndex(Base):
    __tablename__ = "faiss_index"
    id = Column(Integer, primary_key=True, index=True)
    index_data = Column(LargeBinary)  # Store the serialized FAISS index
    created_at = Column(Date, default=datetime.utcnow)
    document_type = Column(String(1), nullable=False) 


# 3. Database Setup
DATABASE_URL = "sqlite+aiosqlite:///./employee.db"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# @event.listens_for(engine.sync_engine, "connect")
# def sqlite_connect(dbapi_connection, connection_record):
#     def parse_datetime(val):
#         return datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
    
#     dbapi_connection.create_function("datetime", 1, parse_datetime)

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

@app.on_event("shutdown")
async def shutdown_event():
    async with AsyncSessionLocal() as db:
        try:
            # Save all in-memory indices back to DB
            for doc_type, store in vector_stores.items():
                if store:
                    serialized = pickle.dumps(store)
                    db_index = FAISSIndex(
                        index_data=serialized,
                        document_type=doc_type
                    )
                    db.add(db_index)
            await db.commit()
        except Exception as e:
            logger.error(f"Error saving indices on shutdown: {str(e)}")

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
    """Simplified to only return text chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

async def process_file(file: UploadFile) -> List[Document]:
    """Returns documents with guaranteed unique IDs"""
    file_path = await save_file(file)
    try:
        text = extract_text_from_file(file_path)
        file_hash = hashlib.md5(text.encode()).hexdigest()[:8]  # First 8 chars of MD5 hash
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        documents = []
        for i, chunk in enumerate(splitter.split_text(text)):
            # Create ID with: file_hash + chunk_idx + timestamp + random suffix
            doc_id = f"{file_hash}-{i}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": file.filename,
                    "doc_id": doc_id,  # Globally unique ID
                    "doc_type": "text_chunk",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ))
        return documents
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

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

def get_conversational_chain(document_types: List[str]):
    """Create a conversation chain that handles optional chat history"""
    relevant_stores = [vector_stores[t] for t in document_types if vector_stores[t] is not None]
    
    if not relevant_stores:
        raise HTTPException(status_code=400, detail="No documents available for your access level")
    
    # Merge stores if needed
    vector_store = relevant_stores[0]
    if len(relevant_stores) > 1:
        for store in relevant_stores[1:]:
            vector_store.merge_from(store)
    
    llm = get_llm_with_retry()
    retriever = vector_store.as_retriever()
    
    # Initialize memory only if needed
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=True
    )

# 6. FAISS Index Management
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

async def save_faiss_index_to_db(documents: List[Document], document_type: str, db: AsyncSession):
    """Handles document indexing with proper ID management"""
    global vector_stores
    
    try:
        if not documents:
            raise ValueError("No documents provided")

        # Generate unique IDs for all documents
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        
        if vector_stores[document_type] is None:
            logger.info(f"Creating new index for type {document_type}")
            vector_stores[document_type] = FAISS.from_documents(
                documents=documents,
                embedding=embeddings_model,
                ids=doc_ids  # Explicitly provide IDs
            )
        else:
            logger.info(f"Adding {len(documents)} documents to existing index for type {document_type}")
            # First get existing IDs to avoid conflicts
            existing_ids = set(vector_stores[document_type].index_to_docstore_id.values())
            new_ids = set(doc_ids)
            
            if existing_ids.intersection(new_ids):
                logger.warning("Duplicate IDs detected, regenerating...")
                doc_ids = [str(uuid.uuid4()) for _ in documents]
            
            vector_stores[document_type].add_documents(
                documents=documents,
                ids=doc_ids
            )

        # Serialize and save - first clear old index
        await db.execute(
            FAISSIndex.__table__.delete().where(
                FAISSIndex.document_type == document_type
            )
        )
        
        serialized = pickle.dumps(vector_stores[document_type])
        db.add(FAISSIndex(
            index_data=serialized,
            document_type=document_type,
            created_at=datetime.utcnow()
        ))
        await db.commit()
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Index save failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

async def load_faiss_index_from_db(db: AsyncSession):
    global vector_stores
    try:
        result = await db.execute(select(FAISSIndex).order_by(FAISSIndex.created_at.desc()))
        all_indices = result.scalars().all()
        
        # Clear existing stores
        vector_stores = {t: None for t in ['A', 'B', 'C']}
        
        if not all_indices:
            logger.warning("No indices found in database")
            return
        
        for db_index in all_indices:
            try:
                store = pickle.loads(db_index.index_data)
                if store.index.ntotal > 0:  # Only accept non-empty indices
                    vector_stores[db_index.document_type] = store
                    logger.info(f"Loaded {db_index.document_type} index with {store.index.ntotal} embeddings")
            except Exception as e:
                logger.error(f"Failed to load {db_index.document_type} index: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Database error loading indices: {str(e)}")
        raise

async def save_faiss_index(all_text_chunks: List[str], document_type: str, db: AsyncSession):
    """Save the FAISS index to the database for a specific document type."""
    await save_faiss_index_to_db(all_text_chunks, document_type, db)


async def load_faiss_index(db: AsyncSession):
    """Load all FAISS indices from the database."""
    await load_faiss_index_from_db(db)

# 7. API Endpoints

class UploadRequest(BaseModel):
    document_type: str

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    upload_request: UploadRequest = Depends(),
    token: str = Header(None),
    db: AsyncSession = Depends(get_db)
):
    start_time = time.time()

    if token not in VALID_TOKENS or VALID_TOKENS[token] != "upload":
        logger.error(f"Unauthorized access attempt with token: {token}")
        raise HTTPException(status_code=404, detail="Unauthorized")

    # Validate document type
    if upload_request.document_type not in ['A', 'B', 'C']:
        raise HTTPException(status_code=400, detail="Invalid document type. Must be A, B, or C.")

    all_documents = []  # Changed from all_text_chunks to all_documents
    file_processing_times = []

    for file in files:
        file_start_time = time.time()
        try:
            file_size_mb = file.size / (1024 * 1024)
            logger.info(f"Processing file: {file.filename} (Size: {file_size_mb:.2f} MB) for type {upload_request.document_type}")

            # Changed to handle Document objects instead of raw text chunks
            documents = await process_file(file)
            all_documents.extend(documents)

            file_end_time = time.time()
            file_processing_time = file_end_time - file_start_time
            file_processing_times.append(file_processing_time)
            logger.info(f"Processed {len(documents)} chunks from {file.filename} in {file_processing_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")

    total_file_processing_time = time.time() - start_time
    logger.info(f"Total file processing time: {total_file_processing_time:.2f} seconds.")

    faiss_start_time = time.time()
    try:
        # Changed to pass documents instead of text chunks
        await save_faiss_index(all_documents, upload_request.document_type, db)
        if vector_stores[upload_request.document_type] is None:
            raise HTTPException(status_code=500, detail="Failed to create index")
            
        logger.info(f"Index contains {vector_stores[upload_request.document_type].index.ntotal} embeddings")
        await load_faiss_index(db)  # Force immediate reload
        return {
            "message": f"{len(files)} files processed",
            "index_size": vector_stores[upload_request.document_type].index.ntotal,
            "document_type": upload_request.document_type
        }
    except Exception as e:
        logger.error(f"FAISS indexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    faiss_end_time = time.time()
    faiss_indexing_time = faiss_end_time - faiss_start_time
    logger.info(f"FAISS indexing completed in {faiss_indexing_time:.2f} seconds.")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Upload and processing completed in {total_time:.2f} seconds.")

    return {
        "message": f"{len(files)} files uploaded successfully",
        "details": {
            "document_type": upload_request.document_type,
            "total_chunks": len(all_documents),
            "processing_time": f"{total_time:.2f} seconds"
        }
    }

# Query Models
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[str]] = None

def extract_answer(response):
    return response.get("answer", "No answer found.")

@app.post("/query")
async def query_document(data: QueryRequest, employee_type: str = Query(..., title="Employee Type", regex="^[A-C]$"), token: str = Header(None)):
    """
    Example request:
    POST /query?employeeType=B
    Header: Authorization: Bearer qwerty
    Body: {"question": "Tell me about DocuSeek"}
    """
    if token not in VALID_TOKENS or VALID_TOKENS[token] not in ["upload", "query"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        allowed_doc_types = ACCESS_RULES[employee_type]
        available_stores = [vector_stores[t] for t in allowed_doc_types if vector_stores[t] is not None]
        
        if not available_stores:
            raise HTTPException(
                status_code=404,
                detail="No documents available for your access level"
            )

        # Approach: Query each index separately and combine results
        all_results = []
        for store in available_stores:
            try:
                # Get relevant chunks from this index
                docs = store.similarity_search(data.question, k=5)
                all_results.extend(docs)
            except Exception as e:
                logger.error(f"Error searching index: {str(e)}")
                continue

        if not all_results:
            return {"answer": "No relevant information found in available documents"}

        # Create a temporary memory for this query
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        # Create a temporary FAISS index just for this query
        temp_index = FAISS.from_documents(
            documents=all_results,
            embedding=embeddings_model
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=get_llm_with_retry(),
            retriever=temp_index.as_retriever(),
            memory=memory,
            verbose=True
        )
        
        response = chain.invoke({
            "question": data.question,
            "chat_history": data.chat_history or []
        })
        
        return {"answer": response.get("answer", "No answer could be generated")}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faiss-index/")
async def inspect_faiss_index(db: AsyncSession = Depends(get_db)):
    """Inspect all FAISS indices metadata."""
    index_info = {}
    for doc_type, vector_store in vector_stores.items():
        if vector_store is not None:
            index_info[doc_type] = {
                "index_size": vector_store.index.ntotal,
                "index_dimensions": vector_store.index.d,
            }
        else:
            index_info[doc_type] = "No index loaded"
    
    return index_info

@app.delete("/clear-faiss-index/")
async def clear_faiss_index(db: AsyncSession = Depends(get_db)):
    """Clear all rows from the faiss_index table."""
    await db.execute(FAISSIndex.__table__.delete())
    await db.commit()
    # Also clear in-memory stores
    global vector_stores
    vector_stores = {t: None for t in ['A', 'B', 'C']}
    return {"message": "All FAISS indices cleared successfully."}


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application."}

@app.get("/health/")
async def health_check():
    return {"status": "ok"}