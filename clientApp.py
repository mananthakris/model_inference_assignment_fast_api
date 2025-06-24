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
    clientApp = ClientApp()
    decodeImage(payload.image, clientApp.filename)
    return clientApp.classifier.prediction_dog_cat()


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__=="__main__":
    uvicorn.run("clientApp:app", host = "0.0.0.0", port=9000, reload=False)
