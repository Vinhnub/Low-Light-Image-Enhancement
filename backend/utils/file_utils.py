import aiofiles
from fastapi import UploadFile
import os
import base64
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

def get_image_base64(filepath: str) -> str:
    """
    Reads an image from disk and converts it to a Base64 string.
    """
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        # Format it so frontend can use it directly in <img src="..." />
        return f"data:image/png;base64,{encoded_string}"

