# DocuSeek - Document Search and Query System

This is a comprehensive FastAPI-based system for uploading, indexing, and querying documents with access control based on employee types. Let me break down the technical details and workflow.

## System Overview

The system provides:
1. Secure document upload and indexing with access control
2. Vector-based semantic search capabilities
3. Conversational question answering over documents
4. Multi-level access control for different document types

## Key Components

### 1. Core Technologies

- **FastAPI**: Web framework for building the API endpoints
- **SQLAlchemy Async**: For database operations with async support
- **FAISS**: Facebook's vector similarity search library
- **HuggingFace Embeddings**: Sentence transformers for text embeddings
- **Google Gemini**: LLM for generating answers
- **PyPDF2**: For PDF text extraction

### 2. Database Structure

The system uses SQLite with these main tables:
- `faiss_index`: Stores serialized FAISS indices by document type

### 3. Document Types and Access Control

Three document types with hierarchical access:
- Type A: Most restricted (only accessible to A employees)
- Type B: Accessible to A and B employees
- Type C: Accessible to all employees (A, B, C)

## Detailed Workflow

### Upload Process (`/upload` endpoint)

1. **Authentication**:
   - Validates the upload token
   - Checks document type is valid (A, B, or C)

2. **File Processing**:
   - Files are saved temporarily
   - PDFs are read using PyPDF2, text files directly read
   - Each file's text is extracted and split into chunks

3. **Document Preparation**:
   - Each chunk becomes a `Document` object with:
     - Unique ID (file hash + chunk index + timestamp)
     - Metadata including source filename and document type
     - Text content

4. **Vector Indexing**:
   - Documents are embedded using HuggingFace's MiniLM model
   - FAISS index is created/updated for the document type
   - Index is serialized and saved to database

5. **Cleanup**:
   - Temporary files deleted
   - Memory indices updated

### Search Process (`/query` endpoint)

1. **Authentication & Authorization**:
   - Validates query token
   - Determines accessible document types based on employee type

2. **Vector Search**:
   - For each accessible document type with an index:
     - Performs similarity search for the question
     - Retrieves top 5 most relevant chunks

3. **Response Generation**:
   - Creates temporary FAISS index with all relevant chunks
   - Uses Google Gemini to generate an answer based on:
     - Retrieved chunks
     - Any provided chat history
   - Returns the generated answer

## Technical Highlights

### 1. Vector Index Management

- **In-Memory Indices**: Maintains separate FAISS indices for each document type in memory
- **Persistence**: Serializes indices to database on shutdown or when updated
- **Loading**: Loads all indices at startup for fast access

### 2. Document Processing Pipeline

1. File → Text extraction → Chunking → Embedding → Indexing
2. Each chunk gets a globally unique ID based on:
   - File content hash
   - Chunk position
   - Timestamp
   - Random UUID portion

### 3. Search Architecture

- **Multi-Index Query**: Queries each accessible index separately then combines results
- **Dynamic Index Creation**: For each query, creates a temporary index with relevant chunks
- **Conversational Memory**: Supports follow-up questions through chat history

### 4. Performance Considerations

- **Chunking Strategy**: 1000-character chunks with 200-character overlap
- **Retries**: For rate-limited LLM API calls
- **Async Processing**: For file uploads and database operations
- **Caching**: Maintains indices in memory between requests

## Security Features

1. **Token-based Authentication**: Separate tokens for upload and query
2. **Access Control**: Strict document type access rules
3. **Input Validation**: All inputs validated before processing
4. **Error Handling**: Comprehensive error handling and logging

## Scalability Aspects

1. **Database Backend**: Could be switched to PostgreSQL for larger scale
2. **Index Partitioning**: Already supports multiple indices by document type
3. **Async Operations**: Allows concurrent processing
4. **Chunk Management**: Unique IDs prevent collisions during scaling

## Monitoring and Maintenance

1. **Logging**: Detailed logging at each processing step
2. **Health Check**: `/health/` endpoint for monitoring
3. **Index Inspection**: `/faiss-index/` endpoint for debugging
4. **Clear Operation**: Ability to reset all indices via `/clear-faiss-index/`

This system provides a robust foundation for document search and question answering with proper access controls and efficient vector-based retrieval. The architecture allows for horizontal scaling by adding more workers and using a shared database backend.
