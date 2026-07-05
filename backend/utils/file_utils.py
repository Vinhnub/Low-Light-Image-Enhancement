import aiofiles
from fastapi import UploadFile
import os
from dotenv import load_dotenv

load_dotenv()
SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
PORT_TCP = os.getenv("PORT_TCP", "8000")

async def save_upload_file(upload_file: UploadFile, destination: str):
    """
    Asynchronously save an uploaded file to a specific destination.
    """
    async with aiofiles.open(destination, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)

def get_output_url(filename: str) -> str:
    """
    Construct the URL to access the processed output file.
    """
    return f"http://{SERVER_IP}:{PORT_TCP}/outputs/{filename}"
