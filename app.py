import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
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

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your environment variables.")

# Initialize FastAPI
app = FastAPI()

# ---------- Database Setup ----------
DATABASE_URL = "sqlite:///./employee.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phoneNumber = Column(String, unique=True, index=True)
    type = Column(String, index=True)
    role = Column(String, index=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Store uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FAISS Vector Store
vector_store = None  # This will hold our FAISS index

class APIRateLimitError(Exception):
    """Custom exception for handling API rate limits."""
    pass

def get_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def get_text_chunks(text: str) -> List[str]:
    """Split text into chunks for better processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks: List[str]):
    """Convert text chunks into embeddings and store in FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(APIRateLimitError),
)
def get_llm_with_retry():
    """Instantiate the LLM with retry mechanism."""
    try:
        return ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "quota exceeded" in str(e).lower():
            raise APIRateLimitError(f"Rate limit error: {e}")
        raise

def get_conversational_chain(with_memory: bool = True):
    """Creates a conversational retrieval chain with FAISS retriever."""
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    llm = get_llm_with_retry()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if with_memory else None
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs and update the FAISS vector store cumulatively."""
    global vector_store

    all_text_chunks = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = get_pdf_text(file_path)
        text_chunks = get_text_chunks(text)
        all_text_chunks.extend(text_chunks)  # Collect all chunks

    # Convert new chunks to embeddings
    new_vector_store = get_vector_store(all_text_chunks)

    # Merge with existing FAISS index if it exists
    if vector_store is None:
        vector_store = new_vector_store
    else:
        vector_store.merge_from(new_vector_store)  # Merge embeddings

    return {"message": f"{len(files)} files uploaded and processed successfully."}

# ✅ Define request model for JSON input
class QueryRequest(BaseModel):
    question: str

def extract_answer(response):
    """Extract only the answer from the response"""
    return response.get("answer","No answer found.")

@app.post("/query")
async def query_document(data: QueryRequest):
    """Query without conversation memory (stateless query)."""
    chain = get_conversational_chain(with_memory=False)
    response = chain.invoke({"question": data.question,"chat_history": []})
    return {"answer": extract_answer(response)}

@app.post("/chat")
async def chat_with_memory(data: QueryRequest):
    """Chat with memory (stateful conversation)."""
    chain = get_conversational_chain(with_memory=True)
    response = chain.invoke({"question": data.question, "chat_history": []})
    return {"answer": extract_answer(response)}

# ---------- Employee CRUD APIs ----------
class EmployeeCreate(BaseModel):
    name: str
    phoneNumber: str
    type: str
    role: str  # Added role since it was used in Employee model

class EmployeesCreate(BaseModel):
    employees: List[EmployeeCreate]
    class Config:
        orm_mode = True

@app.post("/employees/")
def create_employees(employees_data: EmployeesCreate, db: Session = Depends(get_db)):
    print(employees_data.dict())
    employees_list = [
        Employee(
            name=emp.name,
            phoneNumber=emp.phoneNumber,
            type=emp.type,
            role=emp.role
        ) 
        for emp in employees_data.employees  # Corrected reference
    ]
    
    db.add_all(employees_list)
    db.commit()
    
    return {"message": f"{len(employees_list)} employees added successfully"}


@app.get("/employees/search/")
def search_employee_by_phone(phoneNumber: str, db: Session = Depends(get_db)):
    employee = db.query(Employee).filter(Employee.phoneNumber == phoneNumber).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee

@app.get("/employees/{employee_id}")
def read_employee(employee_id: int, db: Session = Depends(get_db)):
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee

@app.put("/employees/{employee_id}")
def update_employee(employee_id: int, employee: EmployeeCreate, db: Session = Depends(get_db)):
    db_employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    db_employee.name = employee.name
    db_employee.phoneNumber = employee.phoneNumber
    db_employee.type = employee.type
    db.commit()
    db.refresh(db_employee)
    return db_employee

@app.delete("/employees/{employee_id}")
def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    db_employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    db.delete(db_employee)
    db.commit()
    return {"message": "Employee deleted successfully"}