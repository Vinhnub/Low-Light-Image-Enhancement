from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import uvicorn

from backend.routes import image_router

load_dotenv()
SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
PORT_TCP = os.getenv("PORT_TCP", "8000")

app = FastAPI(title="Low-Light Image Enhancement API", version="1.0.0")

# Setup CORS
origins = ["*"]
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
    uvicorn.run("backend.main:app", host=SERVER_IP, port=int(PORT_TCP), reload=True)