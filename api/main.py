from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import uuid
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
from pathlib import Path
import shutil
import tempfile
import re
import fitz 
import docx
from docx.table import Table as DocxTable
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import logging
import base64
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Support Chatbot API", description="API for text and voice-based support chatbot with document retrieval")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Voice chat configuration
MODEL_ID = "gpt-4o-realtime-preview-2024-12-17"
VOICE = "sage"
OPENAI_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
OPENAI_API_URL = "https://api.openai.com/v1/realtime"

# Text chat configuration
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

# Document processing settings
DOCUMENTS_DIR = Path("documents")
VECTOR_STORE_DIR = Path("vector_store")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure directories exist
DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Global variables
vectorstore = None
document_metadata = {}  # Maps document_id to metadata (filename, page_content, etc.)

# Pydantic models
class TextQuery(BaseModel):
    query: str
    
class ChatResponse(BaseModel):
    answer: str
    references: List[Dict[str, Any]]
    
class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    total_pages: int
    tables: Optional[Dict[int, int]] = None  # page_number -> num_tables

class DocumentsResponse(BaseModel):
    documents: List[DocumentMetadata]

# Document processing functions
def extract_text_and_tables_from_pdf(file_path: str) -> Dict[int, Dict]:
    """Extract text, tables, and images from a PDF file."""
    logger.info(f"Extracting content from PDF: {file_path}")
    result = {}
    doc = fitz.open(file_path)
    
    for page_num, page in enumerate(doc):
        logger.info(f"Processing PDF page {page_num+1}/{len(doc)}")
        page_text = page.get_text()
        page_tables = []
        
        # Extract tables - simplified approach, production would use more robust methods
        # For production, consider using libraries like Camelot or Tabula
        tables = page.find_tables()
        if tables and tables.tables:
            logger.info(f"Found {len(tables.tables)} tables on page {page_num+1}")
            for table_idx, table in enumerate(tables.tables):
                logger.info(f"Processing table {table_idx+1} on page {page_num+1}")
                df = pd.DataFrame([[str(cell) if hasattr(cell, 'text') else str(cell) for cell in row] for row in table.cells])
                page_tables.append(df.to_dict())
        
        # Extract images
        page_images = []
        images = page.get_images(full=True)
        logger.info(f"Found {len(images)} images on page {page_num+1}")
        for img_index, img in enumerate(images):
            logger.info(f"Processing image {img_index+1} on page {page_num+1}")
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            # Convert image to base64 for storage
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            page_images.append({
                "image_id": f"img_{page_num}_{img_index}",
                "image_data": image_b64,
                "width": base_image["width"],
                "height": base_image["height"]
            })
        
        result[page_num] = {
            "text": page_text,
            "tables": page_tables,
            "images": page_images
        }
    
    doc.close()
    logger.info(f"Completed extraction from PDF: {file_path}")
    return result

def extract_text_and_tables_from_docx(file_path: str) -> Dict[int, Dict]:
    """Extract text, tables, and images from a DOCX file."""
    logger.info(f"Extracting content from DOCX: {file_path}")
    result = {}
    doc = docx.Document(file_path)
    
    # Process document paragraph by paragraph
    all_text = ""
    page_breaks = [0]  # Start indices of pages
    
    logger.info(f"Processing {len(doc.paragraphs)} paragraphs in DOCX")
    for para_idx, para in enumerate(doc.paragraphs):
        if para_idx % 50 == 0:
            logger.info(f"Processing paragraph {para_idx+1}/{len(doc.paragraphs)}")
        all_text += para.text + "\n"
        if "PAGEBREAK" in para.text.upper():  # Simplistic page break detection
            logger.info(f"Detected page break at paragraph {para_idx+1}")
            page_breaks.append(len(all_text))
    
    # Add end of document as final page break
    page_breaks.append(len(all_text))
    
    # Process tables
    all_tables = []
    logger.info(f"Processing {len(doc.tables)} tables in DOCX")
    for table_idx, table in enumerate(doc.tables):
        logger.info(f"Processing table {table_idx+1}/{len(doc.tables)}")
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        all_tables.append(pd.DataFrame(table_data).to_dict())
    
    # Distribute tables among pages (simplified approach)
    tables_per_page = {}
    if all_tables:
        tables_per_page = {0: all_tables}  # Assign all tables to first page for simplicity
    
    # Create page-based result
    logger.info(f"Creating {len(page_breaks)-1} pages from DOCX")
    for i in range(len(page_breaks) - 1):
        start_idx = page_breaks[i]
        end_idx = page_breaks[i + 1]
        page_text = all_text[start_idx:end_idx]
        
        result[i] = {
            "text": page_text,
            "tables": tables_per_page.get(i, []),
            "images": []  # DOCX image extraction would require additional libraries
        }
    
    logger.info(f"Completed extraction from DOCX: {file_path}")
    return result

async def process_document(file_path: str, filename: str) -> str:
    """Process a document and add it to the vector store."""
    logger.info(f"Starting processing of document: {filename}")
    file_extension = Path(filename).suffix.lower()
    document_id = str(uuid.uuid4())
    
    try:
        # Extract content based on file type
        if file_extension == '.pdf':
            logger.info(f"Processing as PDF: {filename}")
            content = extract_text_and_tables_from_pdf(file_path)
            file_type = "pdf"
        elif file_extension in ['.docx', '.doc']:
            logger.info(f"Processing as DOCX: {filename}")
            content = extract_text_and_tables_from_docx(file_path)
            file_type = "docx"
        else:
            error_msg = f"Unsupported file type: {file_extension}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Store document metadata
        logger.info(f"Storing metadata for document: {filename} (id: {document_id})")
        document_metadata[document_id] = {
            "filename": filename,
            "file_type": file_type,
            "total_pages": len(content),
            "content": content,
            "tables": {page_num: len(page_data.get("tables", [])) for page_num, page_data in content.items()}
        }
        
        # Prepare text chunks for vector store
        texts = []
        metadatas = []
        
        logger.info(f"Creating chunks for {len(content)} pages")
        for page_num, page_data in content.items():
            page_text = page_data["text"]
            # Create chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_text(page_text)
            logger.info(f"Page {page_num+1}: Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_num,
                    "chunk": i,
                    "source": f"{filename}, Page {page_num + 1}"
                })
        
        # Create embeddings and add to vector store
        logger.info(f"Creating embeddings for {len(texts)} text chunks")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        global vectorstore
        if vectorstore is None:
            logger.info("Creating new vector store")
            vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        else:
            logger.info("Adding to existing vector store")
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
        
        # Save vector store for persistence
        logger.info(f"Saving vector store to {VECTOR_STORE_DIR}")
        vectorstore.save_local(str(VECTOR_STORE_DIR))
        logger.info(f"Successfully processed document: {filename} (id: {document_id})")
        
        return document_id
        
    except Exception as e:
        logger.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
        raise

def format_reference(metadata: Dict[str, Any], content: str) -> Dict[str, Any]:
    """Format reference information to be returned to the client."""
    document_id = metadata.get("document_id")
    page = metadata.get("page")
    doc_metadata = document_metadata.get(document_id, {})
    filename = metadata.get("filename", "Unknown")
    
    # Get the file path to the original document
    file_path = DOCUMENTS_DIR / filename
    
    # Read the file as binary and encode as base64
    document_base64 = ""
    if file_path.exists():
        with open(file_path, "rb") as file:
            document_binary = file.read()
            document_base64 = base64.b64encode(document_binary).decode('utf-8')
    
    return {
        "document_id": document_id,
        "filename": filename,
        "file_type": doc_metadata.get("file_type", "unknown"),
        "page": page + 1 if page is not None else 1,
        "file": document_base64,
        "content_preview": content[:200] + "..." if len(content) > 200 else content
    }

async def load_existing_documents():
    """Load and process all documents from the documents directory."""
    logger.info(f"Checking for existing documents in {DOCUMENTS_DIR}")
    
    # Get list of document files
    document_files = list(DOCUMENTS_DIR.glob("**/*"))
    supported_extensions = ['.pdf', '.doc', '.docx']
    document_files = [f for f in document_files if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not document_files:
        logger.info("No existing documents found")
        return
    
    logger.info(f"Found {len(document_files)} existing documents to process")
    
    # Process each document
    for file_path in document_files:
        try:
            logger.info(f"Processing existing document: {file_path.name}")
            await process_document(str(file_path), file_path.name)
        except Exception as e:
            logger.error(f"Error processing existing document {file_path.name}: {str(e)}", exc_info=True)

async def load_existing_vector_store() -> None:
    """Load existing vector store if available."""
    global vectorstore
    
    if VECTOR_STORE_DIR.exists() and list(VECTOR_STORE_DIR.glob("*")):
        try:
            logger.info(f"Attempting to load existing vector store from {VECTOR_STORE_DIR}")
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True)
            logger.info("Successfully loaded existing vector store")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            vectorstore = None
    else:
        logger.info("No existing vector store found")

# API Endpoints

@app.post("/chat", response_model=ChatResponse)
async def text_chat(query: TextQuery):
    """Process a text chat query and return an answer with references."""
    logger.info(f"Chat query received: {query.query}")
    
    if not vectorstore:
        error_msg = "No documents have been uploaded"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        # Retrieve relevant documents
        logger.info("Performing similarity search for query")
        docs = vectorstore.similarity_search(query.query, k=4)
        
        if not docs:
            logger.info("No relevant documents found")
            return ChatResponse(
                answer="I couldn't find any relevant information in the documents.",
                references=[]
            )
        
        logger.info(f"Found {len(docs)} relevant documents")
        
        # Format the context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate an answer using OpenAI
        logger.info("Generating answer with OpenAI")
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        
        system_prompt = """You are a helpful support chatbot. Answer the user's question based ONLY on the following context from company documents. 
        If the information is not in the context, say you don't know. Include specific page numbers and document names in your answer.
        DO NOT make up information that is not in the context."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.query}"}
        ]
        
        response = llm.invoke(messages)
        logger.info("Generated answer from OpenAI")
        
        # Format references
        references = []
        for doc in docs:
            references.append(format_reference(doc.metadata, doc.page_content))
        
        logger.info(f"Returning answer with {len(references)} references")
        return ChatResponse(answer=response.content, references=references)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Voice chat instructions template
DEFAULT_INSTRUCTIONS = """You are an expert support assistant for the company. Follow these rules:
1. Use only information from the company documents
2. If unsure, say you don't know
3. Reference document names and page numbers when possible
4. Keep your answers concise and to the point for voice interaction"""

@app.post("/rtc-connect")
async def connect_rtc(request: Request):
    """Real-time WebRTC connection endpoint for voice chat."""
    logger.info("RTC connection request received")
    global vectorstore
    
    if not vectorstore:
        error_msg = "Please upload documents first"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        client_sdp = await request.body()
        if not client_sdp:
            raise HTTPException(status_code=400, detail="No SDP provided")
        
        client_sdp = client_sdp.decode()
        logger.info("Processing RTC connection with SDP")
        
        # Create dynamically generated context based on top documents
        logger.info("Finding top documents for initial context")
        top_docs = vectorstore.similarity_search("company overview support help", k=2)
        context = "\n".join([f"Document: {doc.metadata.get('filename', 'Unknown')}, "
                           f"Page: {doc.metadata.get('page', 0) + 1}\n"
                           f"{doc.page_content}\n" for doc in top_docs])
        
        # Generate instructions with document context
        instructions = f"{DEFAULT_INSTRUCTIONS}\n\nHere is some initial context from the company documents:\n{context}"
        logger.info("Generated instructions with context for voice assistant")
        
        async with httpx.AsyncClient() as client:
            # Get ephemeral token
            logger.info("Requesting ephemeral token from OpenAI")
            token_res = await client.post(
                OPENAI_SESSION_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": MODEL_ID, 
                    "modalities": ["audio", "text"],
                    "voice": VOICE, 
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en"
                    },
                }
            )
            
            if token_res.status_code != 200:
                error_msg = f"Token request failed with status code {token_res.status_code}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            token_data = token_res.json()
            ephemeral_token = token_data.get('client_secret', {}).get('value', '')
            
            if not ephemeral_token:
                error_msg = "Invalid token response"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            logger.info("Successfully acquired ephemeral token")
            
            # Perform SDP exchange
            logger.info("Performing SDP exchange with OpenAI")
            sdp_res = await client.post(
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {ephemeral_token}",
                    "Content-Type": "application/sdp"
                },
                params={
                    "model": MODEL_ID,
                    "instructions": instructions,
                    "voice": VOICE,
                },
                content=client_sdp
            )
            
            logger.info(f"SDP exchange completed with status code {sdp_res.status_code}")
            
            return Response(
                content=sdp_res.content,
                media_type='application/sdp',
                status_code=sdp_res.status_code
            )
            
    except Exception as e:
        logger.error(f"Error in RTC connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load existing vector store and documents on startup."""
    logger.info("Starting application...")
    
    # First try to load existing vector store
    await load_existing_vector_store()
    
    # Then process all documents in the documents directory
    await load_existing_documents()
    
    logger.info(f"Startup complete. Vector store initialized: {vectorstore is not None}")
    logger.info(f"Total documents loaded: {len(document_metadata)}")

# Main entry
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)