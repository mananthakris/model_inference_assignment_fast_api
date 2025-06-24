import base64
import os
from fastapi import FastAPI, Request
from predict import DogCatClassifier
from pydantic import BaseModel
from typing import Optional
from components.utils import decodeImage
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='templates')

class ImagePayload(BaseModel):
    image: str # base64 string with header

class ClientApp():
     def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = DogCatClassifier(self.filename)


@app.post("/predict")
def predict(payload: ImagePayload):
    client_app = ClientApp()
    decodeImage(payload.image, client_app.filename)
    return client_app.classifier.prediction_dog_cat()


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__=="__main__":
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("clientApp:app", host = "0.0.0.0", port=port, reload=False)
