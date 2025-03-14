import logging
from fasthtml.common import *
from shad4fast import *
from starlette.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
import httpx
import uvicorn

load_dotenv()
if not os.path.exists("static"):
    os.makedirs("static")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app, rt = fast_app(
    pico=False,
    hdrs=(
        ShadHead(tw_cdn=True, theme_handle=False),
        Link(
            rel="stylesheet",
            href="/static/style.css",
            type="text/css"
        ),
        Script(
            src="/static/script.js",
            type="text/javascript",
            defer=True
        ),
    )
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# The backend API URL - make sure this points to your FastAPI backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Company Support Chatbot application...")
    logger.info(f"Backend URL: {BACKEND_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Company Support Chatbot application...")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return Alert(
        AlertTitle("Error"),
        AlertDescription(f"An error occurred: {str(exc)}"),
        variant="destructive",
        cls="mt-4 backdrop-blur-sm glass"
    )

def get_hero_section():
    """Generate the hero section with main headline"""
    return Div(
        Div(
            Div(
                H1("DIGITAL HUB", cls="text-2xl md:text-5xl font-bold tracking-tight text-center text-white"),
            ),
            Div(
                Div(
                    H1(
                        "Transform Your Support Experience With",
                        cls="text-4xl md:text-5xl font-bold tracking-tight text-center text-white"
                    ),
                    H1(
                        Span("AI Support", cls="text-blue-400"), " Chatbots",
                        cls="text-4xl md:text-5xl font-bold tracking-tight text-center text-white"
                    ),
                    cls="space-y-4"
                ),
                P(
                    "Optimize Your Customer Service With Expert AI Chatbots Designed For Company Support, Documentation Retrieval And More",
                    cls="text-center text-gray-400 max-w-3xl mx-auto"
                ),
                Div(
                    Button(
                        "Start Chat", 
                        id="start-chat-btn",
                        variant="custom",
                        cls="text-white border border-blue-700 font-bold py-3 px-8 rounded-full flex items-center transition-all duration-300",
                        onclick="scrollToChat()"
                    ),
                    cls="flex justify-center"
                ),
                cls="my-auto mx-auto px-4 py-20 text-center space-y-10"
            ),
            cls="container flex-1 items-center justify-center"
        ),
        cls="h-screen flex items-center"
    )

def get_chat_interface():
    """Generate the chat interface with document retrieval pane"""
    return Div(
        Div(
            Div(
                Div(
                    Div(
                        ScrollArea(
                            Div(
                                id="chat-messages",
                                cls="flex flex-col space-y-6"
                            ),
                            cls="h-full mx-4 px-1 pb-2"
                        ),
                        Div(
                            Div(
                                Input(
                                    type="text",
                                    id="message-input",
                                    placeholder="Ask your question here...",
                                    cls="w-full bg-blue-900/50 text-white placeholder-blue-300 border border-blue-800 rounded-l-full py-3 px-4",
                                    onkeydown="if (event.key === 'Enter') { event.preventDefault(); sendMessage(); }"
                                ),
                                Button(
                                    Div(
                                        Lucide("send", cls="w-5 h-5"),
                                        cls="flex items-center justify-center"
                                    ),
                                    id="send-button",
                                    variant="custom",
                                    cls="text-white border border-blue-700 font-bold py-2 px-6 rounded-l-full rounded-r-full transition-colors duration-300 absolute right-4 top-0",
                                    onClick="sendChatMessage()"
                                ),
                                Button(
                                    Div(
                                        Lucide("mic", cls="w-5 h-5"),
                                        cls="flex items-center"
                                    ),
                                    id="voice-button",
                                    variant="custom",
                                    cls="text-white border border-blue-700 font-bold py-2 px-6 rounded-r-full rounded-l-full transition-all duration-300",
                                    onClick="toggleVoiceMode()"
                                ),
                                cls="flex w-full rounded-r-full rounded-l-full bg-transparent relative"
                            ),
                            cls="px-4"
                        ),
                        cls="h-[600px] flex flex-col justify-between w-full transition-all duration-300"
                    ),
                    id="chat-container",
                    cls="w-full max-w-5xl transition-all duration-300",
                ),
                Div(
                    id="document-viewer",
                    cls="hidden ml-10 bg-blue-950/30 backdrop-blur-sm rounded-xl border border-blue-800 shadow-xl p-6 overflow-hidden h-[600px] w-full transition-all duration-300"
                ),
                cls="flex w-full justify-center"
            ),
            cls="w-full mx-auto"
        ),
        id="chat-section",
        cls="container min-h-screen mx-auto px-4 py-16"
    )

@rt('/')
def get():
    return (
        Title("Company Support Chatbot"),
        Script(f"window.BACKEND_URL = '{os.getenv('BACKEND_URL')}';"),
        Body(
            get_hero_section(),
            get_chat_interface(),
            cls="min-h-screen bg-blue text-white overflow-x-hidden"
        )
    )

def run_server():
    """Run the server with proper configuration"""
    
    # Changed the port to 8001 to avoid conflict with the backend on 8000
    config = uvicorn.Config(
        "main:app",
        host="127.0.0.1",
        port=8001,
        log_level="info",
        workers=1,
        reload=True,
        timeout_keep_alive=None,
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    run_server()
else:
    serve()