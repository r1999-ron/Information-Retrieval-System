import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String
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
from sqlalchemy.exc import IntegrityError
from langchain.schema import Document

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your environment variables.")

# Initialize FastAPI
app = FastAPI()

# Database Setup
DATABASE_URL = "sqlite+aiosqlite:///./employee.db"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phoneNumber = Column(String, unique=True, index=True)
    type = Column(String, index=True)
    role = Column(String, index=True)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("startup")
async def startup_event():
    await init_db()
    load_faiss_index()  # Load the FAISS index on startup

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# File Upload Directory
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE_MB = 5  # Limit file size to 5MB

# FAISS Vector Store
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None
FAISS_INDEX_PATH = "faiss_index"

class APIRateLimitError(Exception):
    pass

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

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
    return text

def split_text_into_chunks(text: str) -> List[str]:
    """Splits text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

async def process_file(file: UploadFile) -> List[str]:
    """Handles file saving, text extraction, and chunking."""
    # file_size = 0
    # temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")

    # with open(temp_path, "wb") as buffer:
    #     while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
    #         file_size += len(chunk)
    #         buffer.write(chunk)
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit.")

    file_path = await save_file(file)
    text = extract_text_from_pdf(file_path)
    os.remove(file_path)
    return split_text_into_chunks(text)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry=retry_if_exception_type(APIRateLimitError))
def get_llm_with_retry():
    try:
        return ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "quota exceeded" in str(e).lower():
            raise APIRateLimitError(f"Rate limit error: {e}")
        raise

def get_conversational_chain(with_memory: bool = True):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No FAISS index loaded. Please upload documents first.")
    llm = get_llm_with_retry()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if with_memory else None
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)

# File Upload API
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global vector_store
    all_text_chunks = await asyncio.gather(*[process_file(file) for file in files])
    all_text_chunks = [chunk for chunks in all_text_chunks for chunk in chunks]
    
    new_vector_store = FAISS.from_texts(all_text_chunks, embedding=embeddings_model)

    # Merge with existing FAISS index if exists
    if vector_store is None:
        vector_store = new_vector_store
    else:
        vector_store.merge_from(new_vector_store)
    # Save the updated vector store
    save_faiss_index(all_text_chunks)  # Persist the FAISS index to disk
    return {"message": f"{len(files)} files uploaded and processed successfully."}

def save_faiss_index(all_text_chunks):
    """Save the FAISS index to disk."""
    documents = [Document(page_content=text) for text in all_text_chunks]
    global vector_store
    vector_store = FAISS.from_documents(documents, embeddings_model)
    if vector_store:
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    else:
        print("Vector store creation failed. FAISS index not saved.")

def load_faiss_index():
    """Load the FAISS index from disk."""
    global vector_store
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
        print(f"FAISS index loaded from {FAISS_INDEX_PATH}")
    else:
        print(f"No FAISS index found, starting fresh.")

# Query Models
class QueryRequest(BaseModel):
    question: str
    chat_history: List[str] = []

def extract_answer(response):
    return response.get("answer", "No answer found.")

@app.post("/query")
async def query_document(data: QueryRequest):
    return {"answer": extract_answer(get_conversational_chain(with_memory=False).invoke(vars(data)))}

@app.post("/chat")
async def chat_with_memory(data: QueryRequest):
    return {"answer": extract_answer(get_conversational_chain(with_memory=True).invoke(vars(data)))}

# Employee CRUD APIs
class EmployeeCreate(BaseModel):
    name: str
    phoneNumber: str
    type: str
    role: str

class EmployeesCreate(BaseModel):
    employees: List[EmployeeCreate]
    class Config:
        orm_mode = True

@app.post("/add_employee/")
async def create_employees(employees_data: EmployeesCreate, db: AsyncSession = Depends(get_db)):
    employees_to_insert = []

    for emp in employees_data.employees:
        result = await db.execute(select(Employee).filter(Employee.phoneNumber == emp.phoneNumber))
        existing_employee = result.scalars().first()

        if existing_employee:
            existing_employee.name = emp.name
            existing_employee.type = emp.type
            existing_employee.role = emp.role
        else:
            employees_to_insert.append(Employee(**emp.dict()))

    try:
        if employees_to_insert:
            db.add_all(employees_to_insert)
        await db.commit()
        return {"message": f"{len(employees_to_insert)} new employees added, existing employees updated"}
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Duplicate phone number found!")

@app.get("/employees/")
async def get_all_employees(db: AsyncSession = Depends(get_db), skip: int = 0, limit: int = 10):
    result = await db.execute(select(Employee).offset(skip).limit(limit))
    employees = result.scalars().all()
    
    if not employees:
        raise HTTPException(status_code=404, detail="No employees found")
    
    return employees

@app.get("/employees/search/")
async def search_employee_by_phone(phoneNumber: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).filter(Employee.phoneNumber == phoneNumber))
    employee = result.scalars().first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee

@app.get("/employees/{employee_id}")
async def read_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).filter(Employee.id == employee_id))
    employee = result.scalars().first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee

@app.put("/employees/{employee_id}")
async def update_employee(employee_id: int, employee: EmployeeCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).filter(Employee.id == employee_id))
    db_employee = result.scalars().first()

    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    for key, value in employee.dict().items():
        setattr(db_employee, key, value)

    await db.commit()
    return db_employee

@app.delete("/employees/{employee_id}")
async def delete_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).filter(Employee.id == employee_id))
    db_employee = result.scalars().first()

    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    await db.delete(db_employee)
    await db.commit()
    return {"message": f"Employee with id {employee_id} deleted successfully"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application."}