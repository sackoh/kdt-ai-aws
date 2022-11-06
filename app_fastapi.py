from typing import List, Dict, Union
from fastapi import FastAPI
from pydantic import BaseModel
from model import MLModelHandler, DLModelHandler
import uvicorn

app = FastAPI()

# assign model handler as global variable [2 LINES]


# define request/response data type for validation
class RequestModel(BaseModel):
    text: Union[str, List[str]]
    model_type: str
    use_gpu: Union[bool, None]

class ResponseModel(BaseModel):
    # prediction: {"label":"negative", "score":0.9752}
    prediction: Dict


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict", response_model=ResponseModel)
async def predict(request: RequestModel):
    text = request.text
    text = [text] if isinstance(text, str) else text
    model_type = request.model_type
    use_gpu = request.use_gpu

    # model inference [2 LINES]
    if model_type == 'ml' or not use_gpu:
        ...
    else:
        ...

    # response
    result = {str(i): {'text': t, 'label': l, 'confidence': c}
                         for i, (t, l, c) in enumerate(zip(text, predictions[0], predictions[1]))}

    return ResponseModel(prediction=result)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
