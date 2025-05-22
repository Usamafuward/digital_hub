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
import base64
import traceback
from openai import OpenAI
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

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_ID = "gpt-4o-realtime-preview-2024-12-17"
VOICE = "sage"
OPENAI_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
OPENAI_API_URL = "https://api.openai.com/v1/realtime"


EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

DOCUMENTS_DIR = Path("documents")
VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_ID = "vs_682d690a20d48191aef3b1fd3cc5cbf3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHARS_PER_PAGE = 3000

DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

vectorstore = None
document_metadata = {}

class TextQuery(BaseModel):
    query: str
    
class ChatTranscript(BaseModel):
    question: str
    answer: str
    
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

class VectorSearchRequest(BaseModel):
    query: str

def clean_text(text: str) -> str:
    """Clean extracted text to improve quality and reduce noise."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common header/footer patterns like "King & Wood Mallesons"
    text = re.sub(r'King\s*&\s*Wood\s*Mallesons.*?(?:Version|Date):\s*[\d\.]+.*?(?:\d{2}/\d{2}/\d{4})', '', text, flags=re.IGNORECASE)
    
    # Remove page numbers
    text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
    
    # Remove "User Guide" and similar document identifiers
    text = re.sub(r'User\s+Guide', '', text, flags=re.IGNORECASE)
    
    # Remove Digital Hub Mobile App and related text
    text = re.sub(r'Digital\s+Hub\s+Mobile\s+App.*?Setting\s+Up', '', text, flags=re.IGNORECASE)
    
    # Remove image descriptions (often in brackets or with "Figure" prefix)
    text = re.sub(r'\[.*?\]', '', text)  # Text in square brackets
    text = re.sub(r'Figure\s+\d+:.*?\.', '', text)  # Figure captions
    
    # Remove URLs that might appear in footers
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text.strip()

def extract_text_from_pdf(file_path: str) -> Dict[int, Dict]:
    """Extract text from a PDF file maintaining original page numbers while removing headers/footers."""
    print(f"Extracting text from PDF: {file_path}")
    result = {}
    
    try:
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            page = doc[page_num]
            
            # Get text blocks to analyze layout
            blocks = page.get_text("blocks")
            
            # Skip potential headers (top 10% of page) and footers (bottom 10% of page)
            filtered_blocks = []
            page_height = page.rect.height
            
            for block in blocks:
                y_pos = block[1]  # Top y-coordinate of the block
                
                # Skip blocks that are likely headers or footers
                if y_pos < page_height * 0.1 or y_pos > page_height * 0.9:
                    # Check if the block contains header/footer patterns
                    if re.search(r'King\s*&\s*Wood|Version|User Guide|Digital\s+Hub|Page \d+|\d+ of \d+', block[4]):
                        continue
                
                filtered_blocks.append(block[4])  # Add block text
            
            # Join the filtered content
            page_text = " ".join(filtered_blocks)
            cleaned_text = clean_text(page_text)
            
            # Only store pages with meaningful content
            if len(cleaned_text) > 5:  # Minimum content threshold
                result[page_num] = {
                    "text": cleaned_text,
                    "original_page": page_num
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
    
    try:
        simple_messages = ["hi", "hii", "hello", "hey", "bye", "goodbye", "thanks", "thank you", "kilogram", "weight", "dimensions", "centimeters", "city", "street", "country", "handling", "Understood"]
        if query.query.lower().strip() in simple_messages:
            print("Simple greeting/farewell detected - processing without references")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful support chatbot. Respond naturally to this greeting."},
                    {"role": "user", "content": query.query}
                ],
                temperature=0
            )
            return ChatResponse(answer=response.choices[0].message.content, references=[])
        
        docs_with_scores = vectorstore.similarity_search_with_score(query.query, k=3 )
        docs = [doc for doc, score in docs_with_scores if score > 0.7]

        # Use OpenAI's file search tool
        response = client.responses.create(
            model=CHAT_MODEL,
            input=f"Answer this question based only on the information in the uploaded documents: {query.query}",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID]
            }]
        )
        
        # Extract answer and file citations
        answer = ""
        references = []
        
        for item in response.output:
            if item.type == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        answer = content_item.text
        
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
        
        if any(phrase in answer.lower() for phrase in no_info_phrases):
            print("Response indicates no relevant information - returning without references")
            return ChatResponse(answer=answer, references=[])
        
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
            
        print(f"Returning answer with {len(references)} unique file references")
        return ChatResponse(answer=answer, references=unique_references)
    
    except Exception as e:
        print(f"Error in text chat: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/rtc-connect")
async def book_connect_rtc(request: Request):
    """Unified WebRTC endpoint with dynamic intent routing"""
    print("Unified RTC connection request received")
    global vectorstore
    global document_metadata

    # Common setup for both flows
    client_sdp = await request.body()
    if not client_sdp:
        raise HTTPException(status_code=400, detail="No SDP provided")
    
    client_sdp = client_sdp.decode()

    context_summary = []
    for doc_id, doc_info in document_metadata.items():

        content = doc_info.get("content", {})
        for page_num in sorted(content.keys())[:]:
            page_text = content[page_num].get("text", "")
            if page_text:
                preview = page_text[:355]
                context_summary.append(preview)
        
    context = "\n".join(context_summary)
        
    print(f"Providing document summary context of length: {len(context)}")
    
    instructions = f"""You are a versatile company support assistant that can handle both general support questions and booking requests.

DETERMINE USER INTENT:
- When the user explicitly says they want to create a booking with phrases like: "I want to create a booking", "I need to book a shipment", "I want to send a package", "I need to arrange a pickup" - use the BOOKING WORKFLOW.
- When the user asks about HOW to create a booking with phrases like: "how do I create a booking", "how to book a shipment", "how to add a booking" - use the GENERAL SUPPORT workflow to explain the process using information from documents
- For all other queries - use the GENERAL SUPPORT workflow with document search.

GENERAL SUPPORT WORKFLOW:
1. Use only information from the company documents
2. If you don't find relevant information, politely say you don't know
3. Keep responses concise and clear for voice interaction

BOOKING WORKFLOW:
1. Collect information in this exact order:
   - Full package sender address (street, city, country)
   - Full package receiver address (street, city, country)
   - Package or document description
   - package Weight in kilograms
   - package Dimensions (length × width × height in cm)
   - package Special handling requirements
2. Ask if package return pickup is needed (yes/no)
3. Ask one question at a time, moving to the next after receiving an answer
4. After collecting all information:
   - Generate a single 6-digit booking number
   - Display all collected details in this format:
     Booking number: [number]
     Sender address: [sender address]
     Receiver address: [receiver address]
     Package description: [description]
     Weight: [weight] kg
     Dimensions: [dimensions] cm
     Special handling: [requirements]
     Return pickup: [yes/no]
5. Ask for final confirmation before completing the booking
6. If the user wants to correct details, update only the specified part

Available documents context:
{context}

Remember to stay conversational and helpful throughout the interaction."""
    
    try:
        async with httpx.AsyncClient() as client:
            # Get session token
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
                raise HTTPException(status_code=500, detail="Token request failed")

            token_data = token_res.json()
            ephemeral_token = token_data.get('client_secret', {}).get('value', '')

            # Dynamic parameter configuration
            params = {
                "model": MODEL_ID,
                "instructions": instructions,
                "voice": VOICE,
            }

            # Single SDP exchange with dynamic instructions
            sdp_res = await client.post(
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {ephemeral_token}",
                    "Content-Type": "application/sdp"
                },
                params=params,
                content=client_sdp
            )

            return Response(
                content=sdp_res.content,
                media_type='application/sdp',
                status_code=sdp_res.status_code
            )

    except Exception as e:
        return Response(
            status_code=500,
            content={"detail": f"Unified assistant error: {str(e)}"}
        )

@app.post("/vector_store_search")
async def vector_store_search(request: VectorSearchRequest):
    try:
        # Now we access the query from the request object
        query = request.query
        
        response = client.responses.create(
            model=CHAT_MODEL,
            input=query,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID]
            }]
        )

        return response.model_dump()["output"][1]["content"][0]["text"]

    except Exception as e:
        print(f"Error in vector store search: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
     
@app.post("/references_for_query", response_model=ChatResponse)
async def references_for_query(transcript: ChatTranscript):
    """Retrieve references for a given text query."""
    print(f"Query received: {transcript.question}")
    print(f"Response received: {transcript.answer}")
    
    if not vectorstore:
        error_msg = "No documents have been uploaded"
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        # Simple greetings and thanks in query that don't need references
        simple_query_phrases = [
            "hii", "hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you"
        ]
        
        if any(phrase == transcript.question.lower().strip() or 
               f" {phrase} " in f" {transcript.question.lower().strip()} " 
               for phrase in simple_query_phrases):
            print("Simple greeting query - returning without references")
            return ChatResponse(answer=transcript.question, references=[])
        
        # Complete phrases that indicate no relevant information in the answer
        no_info_phrases = [
            "don't know", 
            "couldn't find", 
            "no information", 
            "not mentioned", 
            "not available",
            "not in the context",
            "not found in the documents",
            "does not contain information"
        ]
        
        # Full phrases that should not trigger references
        exclude_full_phrases = [
            "booking confirmed", "booking has been confirmed",
            "booking number:", 
            "booking details",
            "special handling", 
            "return pickup", 
            "package sender address",
            "package receiver address",
            "package dimensions",
            "street, city, and country?",
            "length, width, and height?", "length width height", "length, width, and height",
            "package description", 
            "kilograms?", "in kilograms."
            "centimeters?", "in centimeters.",
            "thank you",
            "summary:", "summary:",
            "welcome!", "welcome",
        ]
        
        # Check if any no_info_phrases are in the answer
        if any(phrase in transcript.answer.lower() for phrase in no_info_phrases):
            print("Response indicates no relevant information - returning without references")
            return ChatResponse(answer=transcript.question, references=[])
        
        # Check for complete phrases to exclude
        if any(f" {phrase} " in f" {transcript.answer.lower()} " for phrase in exclude_full_phrases):
            print("Response contains specific phrases that don't need references")
            return ChatResponse(answer=transcript.question, references=[])
        
        # Proceed with retrieving references
        docs_with_scores = vectorstore.similarity_search_with_score(transcript.question, k=10)
        docs = [doc for doc, score in docs_with_scores if score > 0.7]
        
        if not docs:
            print("No relevant documents found")
            return ChatResponse(answer=transcript.question, references=[])
        
        print(f"Found {len(docs)} relevant documents")
        
        unique_files = {}
        
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown")

            if filename not in unique_files:
                formatted_ref = format_reference(doc.metadata, doc.page_content)
                unique_files[filename] = formatted_ref

        unique_references = list(unique_files.values())
        
        return ChatResponse(answer=transcript.question, references=unique_references)
    
    except Exception as e:
        print(f"Error in references_for_query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load existing vector store and documents on startup."""
    print("Starting application...")
    await load_existing_vector_store()
    await load_existing_documents()
    print(f"Startup complete. Vector store initialized: {vectorstore is not None}")
    print(f"Total documents loaded: {len(document_metadata)}")
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)