# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# class APIRateLimitError(Exception):
#     """Custom exception for handling API rate limits."""
#     pass

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if GOOGLE_API_KEY is None:
#     raise ValueError("GOOGLE_API_KEY is not set. Please check your environment variables.")

# os.environ['GOOGLE_API_KEY'] =  GOOGLE_API_KEY

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Increased overlap for context
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Fix embedding function
#     return vector_store

# @retry(
#     stop=stop_after_attempt(3),  # Retry up to 3 times
#     wait=wait_exponential(multiplier=2, min=2, max=10),  # Exponential backoff (2s, 4s, 8s, etc.)
#     retry=retry_if_exception_type(APIRateLimitError),  # Retry only on API-specific failures
# )
# def get_llm_with_retry():
#     """Function to instantiate LLM with retry mecahnism."""
#     try:
#         return ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
#     except Exception as e:
#         if "rate_limit" in str(e).lower() or "quota exceeded" in str(e).lower():
#             raise APIRateLimitError(f"Rate limit error: {e}")
#         raise

# def get_conversational_chain(vector_store):
#     """Creates a conversational retrieval chain with a retry mechanism for LLM instantiation."""
#     llm = get_llm_with_retry()
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm, 
#         retriever=vector_store.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain