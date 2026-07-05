from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from backend.controllers import image_controller

router = APIRouter(
    prefix="/enhance",
    tags=["Image Enhancement"]
)

@router.post("/")
async def enhance_image(
    file: UploadFile = File(...),
    model_name: str = Form(default="Retinexformer")
):
    """
    Endpoint to upload a low-light image and select an enhancement model.
    Returns the URL of the enhanced image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        result = await image_controller.process_image(file, model_name)
        return JSONResponse(status_code=200, content={"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
