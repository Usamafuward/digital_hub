from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
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

def extract_text_and_tables_from_pdf(file_path: str) -> Dict[int, Dict]:
    """Extract text, tables, and images from a PDF file."""
    print(f"Extracting content from PDF: {file_path}")
    result = {}
    doc = fitz.open(file_path)
    
    for page_num, page in enumerate(doc):
        print(f"Processing PDF page {page_num+1}/{len(doc)}")
        page_text = page.get_text()
        page_tables = []
        tables = page.find_tables()
        if tables and tables.tables:
            print(f"Found {len(tables.tables)} tables on page {page_num+1}")
            for table_idx, table in enumerate(tables.tables):
                print(f"Processing table {table_idx+1} on page {page_num+1}")
                df = pd.DataFrame([[str(cell) if hasattr(cell, 'text') else str(cell) for cell in row] for row in table.cells])
                page_tables.append(df.to_dict())

        page_images = []
        images = page.get_images(full=True)
        print(f"Found {len(images)} images on page {page_num+1}")
        for img_index, img in enumerate(images):
            print(f"Processing image {img_index+1} on page {page_num+1}")
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
    """Extract text, tables, and images from a DOCX file."""
    print(f"Extracting content from DOCX: {file_path}")
    result = {}
    doc = docx.Document(file_path)

    all_text = ""
    page_breaks = [0]
    
    print(f"Processing {len(doc.paragraphs)} paragraphs in DOCX")
    for para_idx, para in enumerate(doc.paragraphs):
        if para_idx % 50 == 0:
            print(f"Processing paragraph {para_idx+1}/{len(doc.paragraphs)}")
        all_text += para.text + "\n"
        if "PAGEBREAK" in para.text.upper():
            print(f"Detected page break at paragraph {para_idx+1}")
            page_breaks.append(len(all_text))

    page_breaks.append(len(all_text))

    all_tables = []
    print(f"Processing {len(doc.tables)} tables in DOCX")
    for table_idx, table in enumerate(doc.tables):
        print(f"Processing table {table_idx+1}/{len(doc.tables)}")
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        all_tables.append(pd.DataFrame(table_data).to_dict())

    tables_per_page = {}
    if all_tables:
        tables_per_page = {0: all_tables}

    print(f"Creating {len(page_breaks)-1} pages from DOCX")
    for i in range(len(page_breaks) - 1):
        start_idx = page_breaks[i]
        end_idx = page_breaks[i + 1]
        page_text = all_text[start_idx:end_idx]
        
        result[i] = {
            "text": page_text,
            "tables": tables_per_page.get(i, []),
            "images": []
        }
    
    print(f"Completed extraction from DOCX: {file_path}")
    return result

async def process_document(file_path: str, filename: str) -> str:
    """Process a document and add it to the vector store."""
    print(f"Starting processing of document: {filename}")
    file_extension = Path(filename).suffix.lower()
    document_id = str(uuid.uuid4())
    
    try:
        if file_extension == '.pdf':
            print(f"Processing as PDF: {filename}")
            content = extract_text_and_tables_from_pdf(file_path)
            file_type = "pdf"
        elif file_extension in ['.docx', '.doc']:
            print(f"Processing as DOCX: {filename}")
            content = extract_text_and_tables_from_docx(file_path)
            file_type = "docx"
        else:
            error_msg = f"Unsupported file type: {file_extension}"
            print(error_msg)
            raise ValueError(error_msg)

        print(f"Storing metadata for document: {filename} (id: {document_id})")
        document_metadata[document_id] = {
            "filename": filename,
            "file_type": file_type,
            "total_pages": len(content),
            "content": content,
            "tables": {page_num: len(page_data.get("tables", [])) for page_num, page_data in content.items()}
        }

        texts = []
        metadatas = []
        
        print(f"Creating chunks for {len(content)} pages")
        for page_num, page_data in content.items():
            page_text = page_data["text"]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_text(page_text)
            print(f"Page {page_num+1}: Created {len(chunks)} chunks")
            
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
        print(f"Successfully processed document: {filename} (id: {document_id})")
        
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
            
    print(f"Formatting reference for {filename}")

    return {
        "filename": filename,
        "file_type": file_type,
        "file": document_base64,
    }

async def load_existing_documents():
    """Load and process all documents from the documents directory."""
    print(f"Checking for existing documents in {DOCUMENTS_DIR}")
    
    document_files = list(DOCUMENTS_DIR.glob("**/*"))
    supported_extensions = ['.pdf', '.doc', '.docx']
    document_files = [f for f in document_files if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not document_files:
        print("No existing documents found")
        return
    
    print(f"Found {len(document_files)} existing documents to process")
    
    for file_path in document_files:
        try:
            print(f"Processing existing document: {file_path.name}")
            await process_document(str(file_path), file_path.name)
        except Exception as e:
            print(f"Error processing existing document {file_path.name}: {str(e)}")

async def load_existing_vector_store() -> None:
    """Load existing vector store if available."""
    global vectorstore
    
    if VECTOR_STORE_DIR.exists() and list(VECTOR_STORE_DIR.glob("*")):
        try:
            print(f"Attempting to load existing vector store from {VECTOR_STORE_DIR}")
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
        print("Performing similarity search for query")
        docs = vectorstore.similarity_search(query.query, k=4)
        
        if not docs:
            print("No relevant documents found")
            return ChatResponse(
                answer="I couldn't find any relevant information in the documents.",
                references=[]
            )
        
        print(f"Found {len(docs)} relevant documents")
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        print("Generating answer with OpenAI")
        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
        
        system_prompt = """You are a helpful support chatbot. Answer the user's question based ONLY on the following context from company documents. 
        If the information is not in the context, say you don't know. Include specific page numbers and document names in your answer.
        DO NOT make up information that is not in the context."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.query}"}
        ]
        
        response = llm.invoke(messages)
        print("Generated answer from OpenAI")

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

@app.post("/rtc-connect")
async def connect_rtc(request: Request):
    """Real-time WebRTC connection endpoint for voice chat."""
    print("RTC connection request received")
    global vectorstore
    
    if not vectorstore:
        error_msg = "Please upload documents first"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        client_sdp = await request.body()
        if not client_sdp:
            raise HTTPException(status_code=400, detail="No SDP provided")
        
        client_sdp = client_sdp.decode()
        print("Processing RTC connection with SDP")
        
        print("Finding top documents for initial context")
        top_docs = vectorstore.similarity_search("company overview support help", k=2)
        context = "\n".join([f"Document: {doc.metadata.get('filename', 'Unknown')}, "
                           f"Page: {doc.metadata.get('page', 0) + 1}\n"
                           f"{doc.page_content}\n" for doc in top_docs])

        instructions = f"{DEFAULT_INSTRUCTIONS}\n\nHere is some initial context from the company documents:\n{context}"
        print("Generated instructions with context for voice assistant")
        
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
            
            print("Successfully acquired ephemeral token")

            print("Performing SDP exchange with OpenAI")
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