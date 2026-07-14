from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import uvicorn
from contextlib import asynccontextmanager

from backend.routes import image_router
from backend.services.model_manager import load_all_models

load_dotenv()
SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
PORT_IP = os.getenv("PORT_IP", "8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models before the server starts accepting requests
    load_all_models()
    yield
    # Clean up on shutdown if needed
    pass

app = FastAPI(title="Low-Light Image Enhancement API", version="1.0.0", lifespan=lifespan)

# Setup CORS
origins = ['*'
    # "http://26.212.75.55:5173",
    # "http://26.253.176.29:5173",
    # "http://192.168.1.5:5173",
    # "http://10.12.96.95:5173",
    # "http://26.198.149.7:3000",
    # "https://your-frontend-domain.com",  # add if deployed later
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists
os.makedirs("mydata/outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="mydata/outputs"), name="outputs")

# Include routes
app.include_router(image_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to Low-Light Image Enhancement API"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=SERVER_IP, port=int(PORT_IP), reload=True, log_config="backend/logging.yaml", log_level="debug")