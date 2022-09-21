from typing import List, Dict, Union
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from model import MLModelHandler, DLModelHandler
import uvicorn

app = FastAPI()

class RequestModelString(BaseModel):
    """
    Example: '안녕하세요'
    """
    text: str
    do_fast: str

class RequestModelStringList(BaseModel):
    """
    Example: ['안녕하세요', '반갑습니다']
    """
    text: List[str]
    do_fast: str

class ResponseModel(BaseModel):
    """
    Example: {"label":"negative", "score":0.9752}
    """
    prediction: Dict


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=ResponseModel)
async def predict(request: Union[RequestModelString, RequestModelStringList], classifier: MLModelHandler = Depends(get_ML_Handler)):
    text = request.text
    text = [text] if isinstance(text, str) else text
    do_fast = request.do_fast

    # model inference [2 LINES]
    if do_fast:
        pass
    else:
        pass

    # response
    result = {str(i): {'text': t, 'label': l, 'confidence': c}
                         for i, (t, l, c) in enumerate(zip(text, predictions[0], predictions[1]))}

    return ResponseModel(prediction=result)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    