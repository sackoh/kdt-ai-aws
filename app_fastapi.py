from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from model import MLModelHandler, get_ML_Handler
import uvicorn


app = FastAPI()

class RequestModel(BaseModel):
    """
    Example: '안녕하세요'
    """
    text: str


class ResponseModel(BaseModel):
    """
    Example: {"label":"negative", "score":0.9752}
    """
    prediction: Dict

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=ResponseModel)
async def predict(request: RequestModel, classifier: MLModelHandler = Depends(get_ML_Handler)):
    prediction = classifier.handle(request.text)
    return ResponseModel(prediction=prediction)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    