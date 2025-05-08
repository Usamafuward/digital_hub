# from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import httpx
# import uuid
# from typing import List, Dict, Any, Optional, Set
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
# import json
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
    
# class WebRTCSession(BaseModel):
#     session_id: str
#     query: Optional[str] = None

# class QueryTranscript(BaseModel):
#     session_id: str
#     transcript: str
    
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: Dict[str, WebSocket] = {}
#         self.session_queries: Dict[str, Set[str]] = {}
        
#     async def connect(self, websocket: WebSocket, session_id: str):
#         await websocket.accept()
#         self.active_connections[session_id] = websocket
#         self.session_queries[session_id] = set()
#         print(f"New WebSocket connection: {session_id}")
        
#     def disconnect(self, session_id: str):
#         if session_id in self.active_connections:
#             del self.active_connections[session_id]
#         if session_id in self.session_queries:
#             del self.session_queries[session_id]
#         print(f"WebSocket disconnected: {session_id}")
        
#     async def send_document_update(self, session_id: str, documents: List[Dict[str, Any]]):
#         if session_id in self.active_connections:
#             await self.active_connections[session_id].send_json({
#                 "type": "document_update",
#                 "documents": documents
#             })
            
#     def add_query(self, session_id: str, query: str):
#         if session_id in self.session_queries:
#             self.session_queries[session_id].add(query)
            
#     def get_queries(self, session_id: str) -> List[str]:
#         return list(self.session_queries.get(session_id, set()))
    
# manager = ConnectionManager()

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
#             page_text = page_data["text"]
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
#         docs = vectorstore.similarity_search(query.query, k=4)
        
#         if not docs:
#             print("No relevant documents found")
#             return ChatResponse(
#                 answer="I couldn't find any relevant information in the documents.",
#                 references=[]
#             )
        
#         print(f"Found {len(docs)} relevant documents")
        
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        
#         system_prompt = """You are a helpful support chatbot. Answer the user's question based ONLY on the following context from company documents.
#         If the information is not in the context, say you don't know. Reference document names and page numbers when possible.

#         DO NOT make up information that is not in the context and don't use bold and ** characters for reply."""
        
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.query}"}
#         ]
        
#         response = llm.invoke(messages)

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
#         print(f"Error processing query: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
# DEFAULT_INSTRUCTIONS = """You are an expert support assistant for the company. Follow these rules:
# 1. Use only information from the company documents
# 2. If unsure, say you don't know
# 3. Reference document names and page numbers when possible
# 4. Keep your answers concise and to the point for voice interaction"""

# @app.websocket("/ws/{session_id}")
# async def websocket_endpoint(websocket: WebSocket, session_id: str):
#     """WebSocket endpoint for real-time document updates during voice chat."""
#     try:
#         await manager.connect(websocket, session_id)
#         while True:
#             data = await websocket.receive_text()
#             try:
#                 message = json.loads(data)
#                 if message.get("type") == "query":
#                     query = message.get("query", "")
#                     if query and len(query) > 3:
#                         if vectorstore:
#                             docs = vectorstore.similarity_search(query, k=3)
#                             documents = []
#                             for doc in docs:
#                                 documents.append({
#                                     "filename": doc.metadata.get("filename", "Unknown"),
#                                     "page": doc.metadata.get("page", 0) + 1,
#                                     "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
#                                     "source": f"{doc.metadata.get('filename', 'Unknown')}, Page {doc.metadata.get('page', 0) + 1}"
#                                 })
              
#                             await manager.send_document_update(session_id, documents)
#                             manager.add_query(session_id, query)
#             except json.JSONDecodeError:
#                 pass
#     except WebSocketDisconnect:
#         manager.disconnect(session_id)

# @app.post("/rtc/start-session")
# async def start_rtc_session(session: WebRTCSession):
#     """Start a new WebRTC session with optional initial query."""
#     if not session.session_id:
#         session.session_id = str(uuid.uuid4())
        
#     initial_context = ""
#     if session.query and vectorstore:
#         docs = vectorstore.similarity_search(session.query, k=3)
#         initial_context = "\n".join([
#             f"Document: {doc.metadata.get('filename', 'Unknown')}, "
#             f"Page: {doc.metadata.get('page', 0) + 1}\n"
#             f"{doc.page_content}\n" 
#             for doc in docs
#         ])
        
#     return {
#         "session_id": session.session_id,
#         "initial_context": initial_context
#     }

# @app.post("/rtc/process-transcript")
# async def process_transcript(query_data: QueryTranscript):
#     """Process a transcript from the voice chat to extract queries and get relevant documents."""
#     if not query_data.session_id or not query_data.transcript:
#         return {"error": "Missing session_id or transcript"}
    
#     try:
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
#         extract_query_prompt = """
#         Extract the main question or query from this transcript of a voice conversation.
#         Return only the most important question or information request.
#         If there are multiple questions, focus on the most recent or most significant one.
        
#         Transcript: {transcript}
#         """
        
#         messages = [
#             {"role": "system", "content": extract_query_prompt.format(transcript=query_data.transcript)}
#         ]
        
#         response = llm.invoke(messages)
#         extracted_query = response.content.strip()
        
#         if vectorstore and extracted_query:
#             docs = vectorstore.similarity_search(extracted_query, k=3)
            
#             documents = []
#             for doc in docs:
#                 documents.append({
#                     "filename": doc.metadata.get("filename", "Unknown"),
#                     "page": doc.metadata.get("page", 0) + 1,
#                     "content": doc.page_content,
#                     "source": f"{doc.metadata.get('filename', 'Unknown')}, Page {doc.metadata.get('page', 0) + 1}"
#                 })

#             if query_data.session_id in manager.active_connections:
#                 await manager.send_document_update(query_data.session_id, documents)
 
#             manager.add_query(query_data.session_id, extracted_query)
            
#             return {
#                 "query": extracted_query,
#                 "documents": documents
#             }
        
#         return {"query": extracted_query, "documents": []}
        
#     except Exception as e:
#         print(f"Error processing transcript: {str(e)}")
#         return {"error": str(e)}

# @app.get("/rtc/session-history/{session_id}")
# async def get_session_history(session_id: str):
#     """Get the history of queries for a specific session."""
#     if not session_id:
#         return {"error": "Missing session_id"}
    
#     queries = manager.get_queries(session_id)
    
#     return {
#         "session_id": session_id,
#         "queries": queries
#     }

# @app.post("/rtc-connect")
# async def connect_rtc(request: Request):
#     """Enhanced real-time WebRTC connection endpoint for voice chat."""
#     print("RTC connection request received")
#     global vectorstore
    
#     if not vectorstore:
#         error_msg = "Please upload documents first"
#         print(error_msg)
#         raise HTTPException(status_code=400, detail=error_msg)
    
#     try:
#         body = await request.body()
#         body_dict = {}
        
#         try:
#             body_str = body.decode()
#             if body_str.startswith("{"):
#                 body_dict = json.loads(body_str)
#                 client_sdp = body_dict.get("sdp", "")
#                 session_id = body_dict.get("session_id", str(uuid.uuid4()))
#                 initial_query = body_dict.get("initial_query", "")
#             else:
#                 client_sdp = body_str
#                 session_id = str(uuid.uuid4())
#                 initial_query = ""
#         except:
#             client_sdp = body.decode()
#             session_id = str(uuid.uuid4())
#             initial_query = ""
        
#         if not client_sdp:
#             raise HTTPException(status_code=400, detail="No SDP provided")

#         query = initial_query if initial_query else "company overview support help"
#         top_docs = vectorstore.similarity_search(query, k=3)
#         context = "\n".join([f"Document: {doc.metadata.get('filename', 'Unknown')}, "
#                            f"Page: {doc.metadata.get('page', 0) + 1}\n"
#                            f"{doc.page_content}\n" for doc in top_docs])

#         instructions = f"""{DEFAULT_INSTRUCTIONS}

# Here is some initial context from the company documents:
# {context}

# Important: As you chat with the user, the system will automatically retrieve relevant documents 
# based on the conversation and make them available to you. If you need specific information, 
# ask clarifying questions and the system will try to find relevant documentation."""
        
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
#                     "session_id": session_id,
#                 },
#                 content=client_sdp
#             )
            
#             print(f"SDP exchange completed with status code {sdp_res.status_code}")
 
#             response = Response(
#                 content=sdp_res.content,
#                 media_type='application/sdp',
#                 status_code=sdp_res.status_code,
#                 headers={"X-Session-ID": session_id}
#             )
            
#             return response
            
#     except Exception as e:
#         print(f"Error in RTC connection: {str(e)}", exc_info=True)
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
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import uuid
from typing import List, Dict, Any, Optional, Set
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
import json
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

DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

vectorstore = None
document_metadata = {}
session_contexts: Dict[str, Dict[str, Any]] = {}

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
    
class WebRTCSession(BaseModel):
    session_id: str
    query: Optional[str] = None

class QueryTranscript(BaseModel):
    session_id: str
    transcript: str
    
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_queries: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_queries[session_id] = set()
        print(f"New WebSocket connection: {session_id}")
        
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_queries:
            del self.session_queries[session_id]
        print(f"WebSocket disconnected: {session_id}")
        
    async def send_document_update(self, session_id: str, documents: List[Dict[str, Any]]):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({
                "type": "document_update",
                "documents": documents
            })
            
    def add_query(self, session_id: str, query: str):
        if session_id in self.session_queries:
            self.session_queries[session_id].add(query)
            
    def get_queries(self, session_id: str) -> List[str]:
        return list(self.session_queries.get(session_id, set()))
    
manager = ConnectionManager()

def extract_text_and_tables_from_pdf(file_path: str) -> Dict[int, Dict]:
    """Extract text, tables, and images from a PDF file."""
    print(f"Extracting content from PDF: {file_path}")
    result = {}
    doc = fitz.open(file_path)
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        page_tables = []
        tables = page.find_tables()
        if tables and tables.tables:
            for table_idx, table in enumerate(tables.tables):
                df = pd.DataFrame([[str(cell) if hasattr(cell, 'text') else str(cell) for cell in row] for row in table.cells])
                page_tables.append(df.to_dict())

        page_images = []
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
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
    print(f"Completed extraction from PDF: {file_path}")
    return result

def extract_text_and_tables_from_docx(file_path: str) -> Dict[int, Dict]:
    """Extract text, tables, and images from a DOCX file with improved page detection."""
    print(f"Extracting content from DOCX: {file_path}")
    result = {}
    doc = docx.Document(file_path)

    CHARS_PER_PAGE = 3000
    
    all_text = ""
    for para in doc.paragraphs:
        all_text += para.text + "\n"

    total_chars = len(all_text)
    num_pages = max(1, total_chars // CHARS_PER_PAGE + (1 if total_chars % CHARS_PER_PAGE > 0 else 0))

    for page_num in range(num_pages):
        start_idx = page_num * CHARS_PER_PAGE
        end_idx = min((page_num + 1) * CHARS_PER_PAGE, total_chars)
        page_text = all_text[start_idx:end_idx]

        tables_for_page = []
        total_tables = len(doc.tables)
        tables_start_idx = (page_num * total_tables) // num_pages
        tables_end_idx = ((page_num + 1) * total_tables) // num_pages
        
        for table_idx in range(tables_start_idx, tables_end_idx):
            if table_idx < total_tables:
                table = doc.tables[table_idx]
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
                tables_for_page.append(pd.DataFrame(table_data).to_dict())
        
        result[page_num] = {
            "text": page_text,
            "tables": tables_for_page,
            "images": []
        }
    
    print(f"Completed extraction from DOCX: {file_path}")
    return result

async def process_document(file_path: str, filename: str) -> str:
    """Process a document and add it to the vector store."""
    print(f"Processing document: {filename}")
    file_extension = Path(filename).suffix.lower()
    document_id = str(uuid.uuid4())
    
    try:
        if file_extension == '.pdf':
            content = extract_text_and_tables_from_pdf(file_path)
            file_type = "pdf"
        elif file_extension in ['.docx', '.doc']:
            content = extract_text_and_tables_from_docx(file_path)
            file_type = "docx"
        else:
            error_msg = f"Unsupported file type: {file_extension}"
            print(error_msg)
            raise ValueError(error_msg)

        document_metadata[document_id] = {
            "filename": filename,
            "file_type": file_type,
            "total_pages": len(content),
            "content": content,
            "tables": {page_num: len(page_data.get("tables", [])) for page_num, page_data in content.items()}
        }

        texts = []
        metadatas = []
        
        for page_num, page_data in content.items():
            page_text = page_data["text"]
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
        raise

def format_reference(metadata: Dict[str, Any], content: str) -> Dict[str, Any]:
    """Format reference information to be returned to the client."""
    document_id = metadata.get("document_id")
    doc_metadata = document_metadata.get(document_id, {})
    file_type = doc_metadata.get("file_type", "unknown")
    filename = metadata.get("filename", "unknown")
    
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
        
def update_session_context(session_id: str, query: str, relevant_docs: List[Dict[str, Any]]):
    if session_id not in session_contexts:
        session_contexts[session_id] = {
            "context": "",
            "instructions": DEFAULT_INSTRUCTIONS
        }

    context = "\n".join([f"Document: {doc['filename']}, Page: {doc['page']}\n{doc['content']}" for doc in relevant_docs])
    session_contexts[session_id]["context"] = context

def get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    return session_contexts.get(session_id)


@app.post("/chat", response_model=ChatResponse)
async def text_chat(query: TextQuery):
    """Process a text chat query and return an answer with references."""
    print(f"Chat query received: {query.query}")
    
    if not vectorstore:
        error_msg = "No documents have been uploaded"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        docs = vectorstore.similarity_search(query.query, k=4)
        
        if not docs:
            print("No relevant documents found")
            return ChatResponse(
                answer="I couldn't find any relevant information in the documents.",
                references=[]
            )
        
        print(f"Found {len(docs)} relevant documents")
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        
        system_prompt = """You are a helpful support chatbot. Answer the user's question based ONLY on the following context from company documents.
        If the information is not in the context, say you don't know. Reference document names and page numbers when possible.

        DO NOT make up information that is not in the context and don't use bold and ** characters for reply."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.query}"}
        ]
        
        response = llm.invoke(messages)

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
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
DEFAULT_INSTRUCTIONS = """You are an expert support assistant for the company. Follow these rules:
1. Use only information from the company documents
2. If unsure, say you don't know
3. Reference document names and page numbers when possible
4. Keep your answers concise and to the point for voice interaction"""

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time document updates during voice chat."""
    try:
        await manager.connect(websocket, session_id)
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "query":
                    query = message.get("query", "")
                    if query and len(query) > 3:
                        if vectorstore:
                            docs = vectorstore.similarity_search(query, k=3)
                            documents = []
                            for doc in docs:
                                documents.append({
                                    "filename": doc.metadata.get("filename", "Unknown"),
                                    "page": doc.metadata.get("page", 0) + 1,
                                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                                    "source": f"{doc.metadata.get('filename', 'Unknown')}, Page {doc.metadata.get('page', 0) + 1}"
                                })
              
                            await manager.send_document_update(session_id, documents)
                            manager.add_query(session_id, query)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.post("/rtc/start-session")
async def start_rtc_session(session: WebRTCSession):
    """Start a new WebRTC session with optional initial query."""
    if not session.session_id:
        session.session_id = str(uuid.uuid4())
        
    initial_context = ""
    if session.query and vectorstore:
        docs = vectorstore.similarity_search(session.query, k=3)
        initial_context = "\n".join([
            f"Document: {doc.metadata.get('filename', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"{doc.page_content}\n" 
            for doc in docs
        ])
        
    return {
        "session_id": session.session_id,
        "initial_context": initial_context
    }

@app.post("/rtc/process-transcript")
async def process_transcript(query_data: QueryTranscript):
    """Process a transcript from the voice chat to extract queries and get relevant documents."""
    if not query_data.session_id or not query_data.transcript:
        return {"error": "Missing session_id or transcript"}
    
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        extract_query_prompt = """
        Extract the main question or query from this transcript of a voice conversation.
        Return only the most important question or information request.
        If there are multiple questions, focus on the most recent or most significant one.
        
        Transcript: {transcript}
        """
        
        messages = [
            {"role": "system", "content": extract_query_prompt.format(transcript=query_data.transcript)}
        ]
        
        response = llm.invoke(messages)
        extracted_query = response.content.strip()
        
        if vectorstore and extracted_query:
            docs = vectorstore.similarity_search(extracted_query, k=3)
            
            documents = []
            for doc in docs:
                documents.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content,
                    "source": f"{doc.metadata.get('filename', 'Unknown')}, Page {doc.metadata.get('page', 0) + 1}"
                })

            # Update session context with relevant documents
            update_session_context(query_data.session_id, extracted_query, documents)

            if query_data.session_id in manager.active_connections:
                await manager.send_document_update(query_data.session_id, documents)
 
            manager.add_query(query_data.session_id, extracted_query)
            
            return {
                "query": extracted_query,
                "documents": documents
            }
        
        return {"query": extracted_query, "documents": []}
        
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        return {"error": str(e)}
    
@app.post("/rtc/update-instructions/{session_id}")
async def update_instructions(session_id: str, new_instructions: str):
    """Update the instructions for a specific session."""
    if not session_id or not new_instructions:
        return {"error": "Missing session_id or new_instructions"}
    
    if session_id in session_contexts:
        session_contexts[session_id]["instructions"] = new_instructions
        return {"status": "Instructions updated successfully"}
    else:
        return {"error": "Session not found"}

@app.get("/rtc/session-history/{session_id}")
async def get_session_history(session_id: str):
    """Get the history of queries for a specific session."""
    if not session_id:
        return {"error": "Missing session_id"}
    
    queries = manager.get_queries(session_id)
    
    return {
        "session_id": session_id,
        "queries": queries
    }

@app.post("/book-rtc-connect")
async def connect_rtc(request: Request):
    """Enhanced real-time WebRTC connection endpoint for voice chat."""
    print("RTC connection request received")
    global vectorstore
    
    if not vectorstore:
        error_msg = "Please upload documents first"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        body = await request.body()
        body_dict = {}
        
        try:
            body_str = body.decode()
            if body_str.startswith("{"):
                body_dict = json.loads(body_str)
                client_sdp = body_dict.get("sdp", "")
                session_id = body_dict.get("session_id", str(uuid.uuid4()))
                initial_query = body_dict.get("initial_query", "")
            else:
                client_sdp = body_str
                session_id = str(uuid.uuid4())
                initial_query = ""
        except:
            client_sdp = body.decode()
            session_id = str(uuid.uuid4())
            initial_query = ""
        
        if not client_sdp:
            raise HTTPException(status_code=400, detail="No SDP provided")

        # Initialize session context
        session_contexts[session_id] = {
            "context": "",
            "instructions": DEFAULT_INSTRUCTIONS
        }

        instructions = DEFAULT_INSTRUCTIONS
        
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
                    "session_id": session_id,
                },
                content=client_sdp
            )
            
            print(f"SDP exchange completed with status code {sdp_res.status_code}")
 
            response = Response(
                content=sdp_res.content,
                media_type='application/sdp',
                status_code=sdp_res.status_code,
                headers={"X-Session-ID": session_id}
            )
            
            return response
            
    except Exception as e:
        print(f"Error in RTC connection: {str(e)}", exc_info=True)
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)