from fastapi import UploadFile
from backend.services import enhancement_service
from backend.utils.file_utils import save_upload_file, get_output_url
import os
import uuid

async def process_image(file: UploadFile, model_name: str) -> dict:
    """
    Controller logic to handle the uploaded image and coordinate the enhancement.
    """
    # Create a unique filename to avoid collisions
    unique_id = str(uuid.uuid4())
    original_filename = f"{unique_id}_{file.filename}"
    
    # Save the original file to a temporary or data directory
    input_dir = "mydata/inputs"
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, original_filename)
    
    await save_upload_file(file, input_path)

    # Process the image using the selected model via service
    output_filename = f"enhanced_{original_filename}"
    output_dir = "mydata/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Call the service layer to do the actual ML inference
    enhancement_service.enhance(input_path, output_path, model_name)

    # Return the URL for the client to access the enhanced image
    output_url = get_output_url(output_filename)
    
    return {
        "original_image": file.filename,
        "model_used": model_name,
        "enhanced_image_url": output_url
    }
