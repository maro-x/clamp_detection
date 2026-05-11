from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io 
from io import BytesIO
from infer import predict_image
from PIL import Image

app = FastAPI()


@app.post("/predict/model_1")
async def predict(file:UploadFile = File(...)):

    img = Image.open(file.file)
    predicted_image = predict_image(img, model = 'model_1')
    buffer = io.BytesIO()
    predicted_image.save(buffer, format= 'PNG')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='image/png')


