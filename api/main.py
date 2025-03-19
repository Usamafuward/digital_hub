# from fastapi import FastAPI, HTTPException, Request, Response
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import httpx
# import uuid
# from typing import List, Dict, Any, Optional
# import uvicorn
# from pathlib import Path
# import fitz 
# import docx
# import pandas as pd
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI
# import base64
# import traceback
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI(title="Support Chatbot API", description="API for text and voice-based support chatbot with document retrieval")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     print("OPENAI_API_KEY not found in environment variables")

# MODEL_ID = "gpt-4o-realtime-preview-2024-12-17"
# VOICE = "sage"
# OPENAI_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
# OPENAI_API_URL = "https://api.openai.com/v1/realtime"

# EMBEDDING_MODEL = "text-embedding-3-large"
# CHAT_MODEL = "gpt-4o"

# DOCUMENTS_DIR = Path("documents")
# VECTOR_STORE_DIR = Path("vector_store")
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# DOCUMENTS_DIR.mkdir(exist_ok=True)
# VECTOR_STORE_DIR.mkdir(exist_ok=True)

# vectorstore = None
# document_metadata = {}

# class TextQuery(BaseModel):
#     query: str
    
# class ChatResponse(BaseModel):
#     answer: str
#     references: List[Dict[str, Any]]
    
# class DocumentMetadata(BaseModel):
#     document_id: str
#     filename: str
#     file_type: str
#     total_pages: int
#     tables: Optional[Dict[int, int]] = None

# class DocumentsResponse(BaseModel):
#     documents: List[DocumentMetadata]

# def extract_text_and_tables_from_pdf(file_path: str) -> Dict[int, Dict]:
#     """Extract text, tables, and images from a PDF file."""
#     print(f"Extracting content from PDF: {file_path}")
#     result = {}
#     doc = fitz.open(file_path)
    
#     for page_num, page in enumerate(doc):
#         page_text = page.get_text()
#         page_tables = []
#         tables = page.find_tables()
#         if tables and tables.tables:
#             for table_idx, table in enumerate(tables.tables):
#                 df = pd.DataFrame([[str(cell) if hasattr(cell, 'text') else str(cell) for cell in row] for row in table.cells])
#                 page_tables.append(df.to_dict())

#         page_images = []
#         images = page.get_images(full=True)
#         for img_index, img in enumerate(images):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_data = base_image["image"]
#             image_b64 = base64.b64encode(image_data).decode('utf-8')
#             page_images.append({
#                 "image_id": f"img_{page_num}_{img_index}",
#                 "image_data": image_b64,
#                 "width": base_image["width"],
#                 "height": base_image["height"]
#             })
        
#         result[page_num] = {
#             "text": page_text,
#             "tables": page_tables,
#             "images": page_images
#         }
    
#     doc.close()
#     print(f"Completed extraction from PDF: {file_path}")
#     return result

# def extract_text_and_tables_from_docx(file_path: str) -> Dict[int, Dict]:
#     """Extract text, tables, and images from a DOCX file with improved page detection."""
#     print(f"Extracting content from DOCX: {file_path}")
#     result = {}
#     doc = docx.Document(file_path)

#     CHARS_PER_PAGE = 3000
    
#     all_text = ""
#     for para in doc.paragraphs:
#         all_text += para.text + "\n"

#     total_chars = len(all_text)
#     num_pages = max(1, total_chars // CHARS_PER_PAGE + (1 if total_chars % CHARS_PER_PAGE > 0 else 0))

#     for page_num in range(num_pages):
#         start_idx = page_num * CHARS_PER_PAGE
#         end_idx = min((page_num + 1) * CHARS_PER_PAGE, total_chars)
#         page_text = all_text[start_idx:end_idx]

#         tables_for_page = []
#         total_tables = len(doc.tables)
#         tables_start_idx = (page_num * total_tables) // num_pages
#         tables_end_idx = ((page_num + 1) * total_tables) // num_pages
        
#         for table_idx in range(tables_start_idx, tables_end_idx):
#             if table_idx < total_tables:
#                 table = doc.tables[table_idx]
#                 table_data = []
#                 for row in table.rows:
#                     row_data = [cell.text for cell in row.cells]
#                     table_data.append(row_data)
#                 tables_for_page.append(pd.DataFrame(table_data).to_dict())
        
#         result[page_num] = {
#             "text": page_text,
#             "tables": tables_for_page,
#             "images": []
#         }
    
#     print(f"Completed extraction from DOCX: {file_path}")
#     return result

# async def process_document(file_path: str, filename: str) -> str:
#     """Process a document and add it to the vector store."""
#     print(f"Processing document: {filename}")
#     file_extension = Path(filename).suffix.lower()
#     document_id = str(uuid.uuid4())
    
#     try:
#         if file_extension == '.pdf':
#             content = extract_text_and_tables_from_pdf(file_path)
#             file_type = "pdf"
#         elif file_extension in ['.docx', '.doc']:
#             content = extract_text_and_tables_from_docx(file_path)
#             file_type = "docx"
#         else:
#             error_msg = f"Unsupported file type: {file_extension}"
#             print(error_msg)
#             raise ValueError(error_msg)

#         document_metadata[document_id] = {
#             "filename": filename,
#             "file_type": file_type,
#             "total_pages": len(content),
#             "content": content,
#             "tables": {page_num: len(page_data.get("tables", [])) for page_num, page_data in content.items()}
#         }

#         texts = []
#         metadatas = []
        
#         for page_num, page_data in content.items():
#             page_text = page_data["text"].lower()
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=CHUNK_SIZE,
#                 chunk_overlap=CHUNK_OVERLAP
#             )
#             chunks = text_splitter.split_text(page_text)
            
#             for i, chunk in enumerate(chunks):
#                 texts.append(chunk)
#                 metadatas.append({
#                     "document_id": document_id,
#                     "filename": filename,
#                     "page": page_num,
#                     "chunk": i,
#                     "source": f"{filename}, Page {page_num + 1}"
#                 })

#         print(f"Creating embeddings for {len(texts)} text chunks")
#         embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
#         global vectorstore
#         if vectorstore is None:
#             print("Creating new vector store")
#             vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
#         else:
#             print("Adding to existing vector store")
#             vectorstore.add_texts(texts=texts, metadatas=metadatas)

#         print(f"Saving vector store to {VECTOR_STORE_DIR}")
#         vectorstore.save_local(str(VECTOR_STORE_DIR))
        
#         return document_id
        
#     except Exception as e:
#         print(f"Error processing document {filename}: {str(e)}")
#         raise

# def format_reference(metadata: Dict[str, Any], content: str) -> Dict[str, Any]:
#     """Format reference information to be returned to the client."""
#     document_id = metadata.get("document_id")
#     doc_metadata = document_metadata.get(document_id, {})
#     file_type = doc_metadata.get("file_type", "unknown")
#     filename = metadata.get("filename", "unknown")
#     page_number = metadata.get("page", 0)
    
#     if file_type == "unknown":
#         file_extension = filename.split('.')[-1].lower()
#         if file_extension:
#             file_type = file_extension

#     file_path = DOCUMENTS_DIR / filename

#     document_base64 = ""
#     if file_path.exists():
#         with open(file_path, "rb") as file:
#             document_binary = file.read()
#             document_base64 = base64.b64encode(document_binary).decode('utf-8')

#     return {
#         "filename": filename,
#         "file_type": file_type,
#         "file": document_base64,
#         "page": page_number,
#     }

# async def load_existing_documents():
#     """Load and process all documents from the documents directory."""
#     print("Checking for existing documents")
    
#     document_files = list(DOCUMENTS_DIR.glob("**/*"))
#     supported_extensions = ['.pdf', '.doc', '.docx']
#     document_files = [f for f in document_files if f.is_file() and f.suffix.lower() in supported_extensions]
    
#     if not document_files:
#         print("No existing documents found")
#         return
    
#     print(f"Found {len(document_files)} existing documents to process")
    
#     for file_path in document_files:
#         try:
#             await process_document(str(file_path), file_path.name)
#         except Exception as e:
#             print(f"Error processing existing document {file_path.name}: {str(e)}")

# async def load_existing_vector_store() -> None:
#     """Load existing vector store if available."""
#     global vectorstore
    
#     if VECTOR_STORE_DIR.exists() and list(VECTOR_STORE_DIR.glob("*")):
#         try:
#             print("Loading existing vector store")
#             embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
#             vectorstore = FAISS.load_local(str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True)
#             print("Successfully loaded existing vector store")
#         except Exception as e:
#             print(f"Error loading vector store: {str(e)}")
#             vectorstore = None
#     else:
#         print("No existing vector store found")


# @app.post("/chat", response_model=ChatResponse)
# async def text_chat(query: TextQuery):
#     """Process a text chat query and return an answer with references."""
#     print(f"Chat query received: {query.query}")
    
#     if not vectorstore:
#         error_msg = "No documents have been uploaded"
#         print(error_msg)
#         raise HTTPException(status_code=400, detail=error_msg)
    
#     try:
#         # Simple check for irrelevant queries
#         irrelevant_queries = ["hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you"]
#         if query.query.lower().strip() in irrelevant_queries or len(query.query.strip()) < 5:
#             print("Detected irrelevant query, responding without references")
#             return ChatResponse(
#                 answer=get_greeting_response(query.query.lower()),
#                 references=[]
#             )
        
#         standardized_query = query.query.lower()
#         docs = vectorstore.similarity_search(standardized_query, k=4)
        
#         # Check if the retrieved documents are relevant
#         if not docs or is_irrelevant_match(standardized_query, docs):
#             print("No relevant documents found or low relevance match")
#             return ChatResponse(
#                 answer="I couldn't find any relevant information in the documents. Is there something else I can help you with?",
#                 references=[]
#             )
        
#         print(f"Found {len(docs)} relevant documents")
        
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        
#         system_prompt = """You are a helpful support chatbot. Answer the user's question based ONLY on the following context from company documents.
#         If the information is not in the context, say you don't know the provided context does not contain information about your question."""
        
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.query}"}
#         ]
        
#         response = llm.invoke(messages)

#         if "don't know" in response.content.lower() or "couldn't find" or "does not contain information" in response.content.lower():
#             print("LLM indicated no relevant information, returning without references")
#             return ChatResponse(answer=response.content, references=[])

#         unique_files = {}
        
#         for doc in docs:
#             filename = doc.metadata.get("filename", "Unknown")

#             if filename not in unique_files:
#                 formatted_ref = format_reference(doc.metadata, doc.page_content)
#                 unique_files[filename] = formatted_ref

#         unique_references = list(unique_files.values())
        
#         print(f"Returning answer with {len(unique_references)} unique file references")
#         return ChatResponse(answer=response.content, references=unique_references)
    
#     except Exception as e:
#         print(f"Error in text chat: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# def get_greeting_response(query: str) -> str:
#     """Return appropriate responses for greetings and short queries."""
#     if query in ["hi", "hello", "hey"]:
#         return "Hello! How can I help you with your questions about our company documents?"
#     elif query in ["bye", "goodbye"]:
#         return "Goodbye! Feel free to ask if you have more questions."
#     elif query in ["thanks", "thank you"]:
#         return "You're welcome! Let me know if you need anything else."
#     else:
#         return "I'm here to help with questions about our company documents. Could you please provide more details?"

# def is_irrelevant_match(query: str, docs: List[Any]) -> bool:
#     """Check if the retrieved documents are actually relevant to the query."""
#     query_terms = set(query.split())
#     if len(query_terms) <= 2:
#         return False

#     stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "like", "of", "is", "are"}
#     query_terms = query_terms - stopwords

#     for doc in docs:
#         doc_text = doc.page_content.lower()
#         matches = sum(1 for term in query_terms if term in doc_text)

#         if matches / len(query_terms) > 0.3:
#             return False
            
#     return True
    
# DEFAULT_INSTRUCTIONS = """You are an expert support assistant for the company. Follow these rules:
# 1. Use only information from the company documents
# 2. If unsure, say you don't know
# 3. Reference document names and page numbers when possible
# 4. Keep your answers concise and to the point for voice interaction"""

# @app.post("/rtc-connect")
# async def connect_rtc(request: Request):
#     """Real-time WebRTC connection endpoint for voice chat."""
#     print("RTC connection request received")
#     global vectorstore
#     global document_metadata
    
#     if not vectorstore:
#         error_msg = "Please upload documents first"
#         print(error_msg)
#         raise HTTPException(status_code=400, detail=error_msg)
    
#     try:
#         client_sdp = await request.body()
#         if not client_sdp:
#             raise HTTPException(status_code=400, detail="No SDP provided")
        
#         client_sdp = client_sdp.decode()

#         context_parts = []
#         for doc_id, doc_info in document_metadata.items():
#             filename = doc_info.get("filename", "Unknown")
#             print(f"Adding document to context: {filename}")

#             for page_num, page_data in doc_info.get("content", {}).items():
#                 page_text = page_data.get("text", "")
#                 if page_text:
#                     context_parts.append(f"Document: {filename}, Page: {page_num + 1}\n{page_text}\n")

#         context = "\n".join(context_parts)
        
#         print(f"Total context length: {len(context)}")
        
#         instructions = f"{DEFAULT_INSTRUCTIONS}\n\nHere is the full context from the company documents:\n{context}"
        
#         async with httpx.AsyncClient() as client:
#             print("Requesting ephemeral token from OpenAI")
#             token_res = await client.post(
#                 OPENAI_SESSION_URL,
#                 headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
#                 json={
#                     "model": MODEL_ID, 
#                     "modalities": ["audio", "text"],
#                     "voice": VOICE, 
#                     "input_audio_format": "pcm16",
#                     "output_audio_format": "pcm16",
#                     "input_audio_transcription": {
#                         "model": "whisper-1",
#                         "language": "en"
#                     },
#                 }
#             )
            
#             if token_res.status_code != 200:
#                 error_msg = f"Token request failed with status code {token_res.status_code}"
#                 print(error_msg)
#                 raise HTTPException(status_code=500, detail=error_msg)
            
#             token_data = token_res.json()
#             ephemeral_token = token_data.get('client_secret', {}).get('value', '')
            
#             if not ephemeral_token:
#                 error_msg = "Invalid token response"
#                 print(error_msg)
#                 raise HTTPException(status_code=500, detail=error_msg)
            
#             sdp_res = await client.post(
#                 OPENAI_API_URL,
#                 headers={
#                     "Authorization": f"Bearer {ephemeral_token}",
#                     "Content-Type": "application/sdp"
#                 },
#                 params={
#                     "model": MODEL_ID,
#                     "instructions": instructions,
#                     "voice": VOICE,
#                 },
#                 content=client_sdp
#             )
            
#             print(f"SDP exchange completed with status code {sdp_res.status_code}")
            
#             return Response(
#                 content=sdp_res.content,
#                 media_type='application/sdp',
#                 status_code=sdp_res.status_code
#             )
            
#     except Exception as e:
#         print(f"Error in RTC connection: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# @app.on_event("startup")
# async def startup_event():
#     """Load existing vector store and documents on startup."""
#     print("Starting application...")
#     await load_existing_vector_store()
#     await load_existing_documents()
#     print(f"Startup complete. Vector store initialized: {vectorstore is not None}")
#     print(f"Total documents loaded: {len(document_metadata)}")
    

# if __name__ == "__main__":
#     uvicorn.run("api.main:app", host="127.0.0.1", port=8001, reload=True)
    
    
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import httpx
import uuid
from typing import List, Dict, Any, Optional
import uvicorn
from pathlib import Path
import fitz  
import docx
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import base64
import traceback
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Support Chatbot API", description="API for text and voice-based support chatbot with document retrieval")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("OPENAI_API_KEY not found in environment variables")

MODEL_ID = "gpt-4o-realtime-preview-2024-12-17"
VOICE = "sage"
OPENAI_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
OPENAI_API_URL = "https://api.openai.com/v1/realtime"

EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

DOCUMENTS_DIR = Path("documents")
VECTOR_STORE_DIR = Path("vector_store")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHARS_PER_PAGE = 3000

DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

vectorstore = None
document_metadata = {}

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
    tables: Optional[Dict[int, int]] = None

class DocumentsResponse(BaseModel):
    documents: List[DocumentMetadata]

def clean_text(text: str) -> str:
    """Clean extracted text to improve quality and reduce noise."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and footer/header artifacts
    text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
    # Remove URLs that might appear in footers
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text.strip()

def extract_text_from_pdf(file_path: str) -> Dict[int, Dict]:
    """Extract text from a PDF file maintaining original page numbers."""
    print(f"Extracting text from PDF: {file_path}")
    result = {}
    
    try:
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            page = doc[page_num]
            page_text = page.get_text("text")
            cleaned_text = clean_text(page_text)
            
            result[page_num] = {
                "text": cleaned_text,
                "original_page": page_num  # Store the original page number directly
            }
                
        doc.close()
        
        print(f"Completed extraction from PDF: {file_path} with {len(result)} pages")
        return result
        
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        traceback.print_exc()
        raise
    

def extract_text_from_docx(file_path: str) -> Dict[int, Dict]:
    """Extract text from a DOCX file with page numbering."""
    print(f"Extracting text from DOCX: {file_path}")
    result = {}
    doc = docx.Document(file_path)
    
    all_text = ""
    for para in doc.paragraphs:
        all_text += clean_text(para.text) + "\n"

    total_chars = len(all_text)
    num_pages = max(1, total_chars // CHARS_PER_PAGE + (1 if total_chars % CHARS_PER_PAGE > 0 else 0))

    for page_num in range(num_pages):
        start_idx = page_num * CHARS_PER_PAGE
        end_idx = min((page_num + 1) * CHARS_PER_PAGE, total_chars)
        page_text = all_text[start_idx:end_idx]
        
        result[page_num] = {
            "text": page_text,
            "original_pages": [page_num]
        }
    
    print(f"Completed extraction from DOCX: {file_path} with {len(result)} standardized pages")
    return result

async def process_document(file_path: str, filename: str) -> str:
    """Process a document and add it to the vector store."""
    print(f"Processing document: {filename}")
    file_extension = Path(filename).suffix.lower()
    document_id = str(uuid.uuid4())
    
    try:
        if file_extension == '.pdf':
            content = extract_text_from_pdf(file_path)
            file_type = "pdf"
        elif file_extension in ['.docx', '.doc']:
            content = extract_text_from_docx(file_path)
            file_type = "docx"
        else:
            error_msg = f"Unsupported file type: {file_extension}"
            print(error_msg)
            raise ValueError(error_msg)

        document_metadata[document_id] = {
            "filename": filename,
            "file_type": file_type,
            "total_pages": len(content),
            "content": content
        }

        texts = []
        metadatas = []
        
        for page_num, page_data in content.items():
            page_text = page_data["text"].lower()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_text(page_text)
            
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_num,
                    "chunk": i,
                    "source": f"{filename}, Page {page_num + 1}"
                })

        print(f"Creating embeddings for {len(texts)} text chunks")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        global vectorstore
        if vectorstore is None:
            print("Creating new vector store")
            vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        else:
            print("Adding to existing vector store")
            vectorstore.add_texts(texts=texts, metadatas=metadatas)

        print(f"Saving vector store to {VECTOR_STORE_DIR}")
        vectorstore.save_local(str(VECTOR_STORE_DIR))
        
        return document_id
        
    except Exception as e:
        print(f"Error processing document {filename}: {str(e)}")
        traceback.print_exc()
        raise

def format_reference(metadata: Dict[str, Any], content: str) -> Dict[str, Any]:
    """Format reference information to be returned to the client."""
    document_id = metadata.get("document_id")
    doc_metadata = document_metadata.get(document_id, {})
    file_type = doc_metadata.get("file_type", "unknown")
    filename = metadata.get("filename", "unknown")
    page_number = metadata.get("page", 0)
    
    if file_type == "unknown":
        file_extension = filename.split('.')[-1].lower()
        if file_extension:
            file_type = file_extension

    file_path = DOCUMENTS_DIR / filename

    document_base64 = ""
    if file_path.exists():
        with open(file_path, "rb") as file:
            document_binary = file.read()
            document_base64 = base64.b64encode(document_binary).decode('utf-8')

    return {
        "filename": filename,
        "file_type": file_type,
        "file": document_base64,
        "page": page_number,
    }

async def load_existing_documents():
    """Load and process all documents from the documents directory."""
    print("Checking for existing documents")
    
    document_files = list(DOCUMENTS_DIR.glob("**/*"))
    supported_extensions = ['.pdf', '.doc', '.docx']
    document_files = [f for f in document_files if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not document_files:
        print("No existing documents found")
        return
    
    print(f"Found {len(document_files)} existing documents to process")
    
    for file_path in document_files:
        try:
            await process_document(str(file_path), file_path.name)
        except Exception as e:
            print(f"Error processing existing document {file_path.name}: {str(e)}")

async def load_existing_vector_store() -> None:
    """Load existing vector store if available."""
    global vectorstore
    
    if VECTOR_STORE_DIR.exists() and list(VECTOR_STORE_DIR.glob("*")):
        try:
            print("Loading existing vector store")
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True)
            print("Successfully loaded existing vector store")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            vectorstore = None
    else:
        print("No existing vector store found")

@app.post("/chat", response_model=ChatResponse)
async def text_chat(query: TextQuery):
    """Process a text chat query and return an answer with references."""
    print(f"Chat query received: {query.query}")
    
    if not vectorstore:
        error_msg = "No documents have been uploaded"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        simple_messages = ["hi", "hii", "hello", "hey", "bye", "goodbye", "thanks", "thank you"]
        if query.query.lower().strip() in simple_messages:
            print("Simple greeting/farewell detected - processing without references")
            llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
            response = llm.invoke([
                {"role": "system", "content": "You are a helpful support chatbot. Respond naturally to this greeting."},
                {"role": "user", "content": query.query}
            ])
            return ChatResponse(answer=response.content, references=[])

        docs_with_scores = vectorstore.similarity_search_with_score(query.query, k=20)
        docs = [doc for doc, score in docs_with_scores if score > 0.7]
        
        if not docs:
            print("No relevant documents found")
            return ChatResponse(
                answer="I couldn't find any relevant information in the documents.",
                references=[]
            )
        
        print(f"Found {len(docs)} relevant documents")

        formatted_contexts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            formatted_contexts.append(f"[{source}]\n{doc.page_content}")
            
        context = "\n\n".join(formatted_contexts)
        
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        
        system_prompt = """You are a helpful support chatbot. Answer the user's question based ONLY on the following context from company documents.
        Each context section is labeled with its source in [brackets].
        
        When answering, prefer to cite specific documents and page numbers like: "According to [Document X, Page Y]..." 
        
        If the information is not in the context, say you don't know the provided context does not contain information about your question.
        DO NOT make up information that is not in the context."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.query}"}
        ]
        
        response = llm.invoke(messages)

        no_info_phrases = [
            "Sorry",
            "I'm sorry",
            "don't know", 
            "couldn't find", 
            "no information", 
            "not mentioned", 
            "not available",
            "not in the context",
            "not found in the documents",
            "does not contain information",
        ]
        
        if any(phrase in response.content.lower() for phrase in no_info_phrases):
            print("Response indicates no relevant information - returning without references")
            return ChatResponse(answer=response.content, references=[])

        files_dict = {}
        
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown")
            if filename not in files_dict:
                files_dict[filename] = []
 
            files_dict[filename].append(doc.metadata)
        
        unique_files = {}
        
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown")

            if filename not in unique_files:
                formatted_ref = format_reference(doc.metadata, doc.page_content)
                unique_files[filename] = formatted_ref

        unique_references = list(unique_files.values())
        
        print(f"Returning answer with {len(unique_references)} unique file references")
        return ChatResponse(answer=response.content, references=unique_references)
    
    except Exception as e:
        print(f"Error in text chat: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
DEFAULT_INSTRUCTIONS = """You are an expert support assistant for the company. Follow these rules:
1. Use only information from the company documents
2. If unsure, say you don't know
3. Reference document names and page numbers when possible
4. Keep your answers concise and to the point for voice interaction"""

@app.post("/rtc-connect")
async def connect_rtc(request: Request):
    """Real-time WebRTC connection endpoint for voice chat."""
    print("RTC connection request received")
    global vectorstore
    global document_metadata
    
    if not vectorstore:
        error_msg = "Please upload documents first"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        client_sdp = await request.body()
        if not client_sdp:
            raise HTTPException(status_code=400, detail="No SDP provided")
        
        client_sdp = client_sdp.decode()

        context_summary = []
        for doc_id, doc_info in document_metadata.items():
            filename = doc_info.get("filename", "Unknown")
            file_type = doc_info.get("file_type", "Unknown")
            total_pages = doc_info.get("total_pages", 0)
            
            context_summary.append(f"Document: {filename} ({file_type.upper()}) - {total_pages} pages")
            
            # Add sample content from first few pages
            content = doc_info.get("content", {})
            for page_num in sorted(content.keys())[:]:
                page_text = content[page_num].get("text", "")
                if page_text:
                    # Get first 200 chars as a preview
                    preview = page_text[:317] + "..." if len(page_text) > 317 else page_text
                    context_summary.append(preview)
        
        context = "\n".join(context_summary)
        
        print(f"Providing document summary context of length: {len(context)}")
        
        # Modified instructions for voice interface
        instructions = f"""{DEFAULT_INSTRUCTIONS}

Important: You'll receive voice queries from users. For each query:
1. You should search the documents in real-time
2. Respond conversationally and concisely
3. Cite specific documents when providing information

Here are the available documents:
{context}

When you don't know something, just say so politely. Never make up information."""
        
        async with httpx.AsyncClient() as client:
            print("Requesting ephemeral token from OpenAI")
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
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            token_data = token_res.json()
            ephemeral_token = token_data.get('client_secret', {}).get('value', '')
            
            if not ephemeral_token:
                error_msg = "Invalid token response"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
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
            
            print(f"SDP exchange completed with status code {sdp_res.status_code}")
            
            return Response(
                content=sdp_res.content,
                media_type='application/sdp',
                status_code=sdp_res.status_code
            )
            
    except Exception as e:
        print(f"Error in RTC connection: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load existing vector store and documents on startup."""
    print("Starting application...")
    await load_existing_vector_store()
    await load_existing_documents()
    print(f"Startup complete. Vector store initialized: {vectorstore is not None}")
    print(f"Total documents loaded: {len(document_metadata)}")
    

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8001, reload=True)