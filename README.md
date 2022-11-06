# K-Digital Training: 프로그래머스 인공지능 데브코스

AWS를 활용한 인공지능 모델 배포 강의 실습
<br><br>
## AWS 실습 개발환경 준비

1. Miniconda를 설치합니다.
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh
    ```
2. 실습 repo를 다운로드합니다.
    ```bash
    sudo yum install git

    git clone https://github.com/sackoh/kdt-ai-aws

    cd kdt-ai-aws
    ```
<br>
## 파이썬 패키지 설치
`requirements.txt`에 있는 실습에 필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
<br>
## 데이터준비 / 모델학습 과정
1. 실습에 사용할 네이버 영화 리뷰 데이터를 다운로드 합니다.
  ([https://github.com/e9t/nsmc](https://github.com/e9t/nsmc))
2. `scikit-learn` 라이브러리를 활용하여 나이브 베이즈 모델을 학습합니다.
3. 학습에 사용하지 않은 테스트 데이터를 통해 모델 성능을 평가합니다.
4. 학습한 모델을 저장합니다.

```bash
python train_ml.py
```

```bash
28-Oct-22 18:54:05 - Downloaded from https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
28-Oct-22 18:54:06 - Downloaded from https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
28-Oct-22 18:54:15 - fitting Counter vectorizer
28-Oct-22 18:54:17 - Transform raw text into vector
28-Oct-22 18:54:17 - Trained Naive Bayes model.
28-Oct-22 18:54:17 - ML model accuracy score: 82.15%
28-Oct-22 18:54:19 - Saved vectorizer to `model/ml_vectorizer.pkl`
28-Oct-22 18:54:19 - Saved model to `model/ml_model.pkl`
28-Oct-22 18:54:19 - Elapsed time : 0:00:17.325072
```
<br>
## 모델 핸들러
- `model.py`
- 학습한 모델을 사용하여 실제로 들어오는 데이터를 추론해내기 위한 핸들러를 개발합니다.
- 모델 로드, 전처리, 추론, 후처리의 데이터 흐름을 직접 만들어봅니다.

```python
class MLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self, ):
        # De-serializing model and loading vectorizer
        ...

    def preprocess(self, ):
        # cleansing raw text
        ...
        # vectorizing cleaned text
        ...


    def inference(self, ):
        # get predictions from model as probabilities
        ...

    def postprocess(self, data):
        # process predictions to predicted label and output format
        ...

    def handle(self, data):
        # do above processes
        ...
```
<br>
## Flask 모델 서빙
- 학습한 모델을 불러와서 `Flask` 프레임워크를 사용하여 서빙합니다.
- `model.py`에서 만든 핸들러를 불러오고, 입력 값으로부터 추론한 결과를 전달하는 코드를 개발합니다.

```python
# assign model handler as global variable [2 LINES]
...


@app.route("/predict", methods=["POST"])
def predict():
    # handle request and body
    body = request.get_json()
    text = body.get('text', '')
    text = [text] if isinstance(text, str) else text
    model_type = body.get('model_type', 'ml')

    # model inference [2 LINES]
    if model_type == 'ml':
        ...
    else:
        ...

    # response
    result = json.dumps({str(i): {'text': t, 'label': l, 'confidence': c}
                         for i, (t, l, c) in enumerate(zip(text, predictions[0], predictions[1]))})
    return result
```

## FastAPI 모델 서빙

- 학습한 모델을 불러와서 `FastAPI` 프레임워크를 사용하여 서빙합니다.
- `model.py`에서 만든 핸들러를 불러오고, 입력 값으로부터 추론한 결과를 전달하는 코드를 개발합니다.

```python
# assign model handler as global variable [2 LINES]
...


@app.post("/predict", response_model=ResponseModel)
async def predict(request: Union[RequestModelString, RequestModelStringList]):
    text = request.text
    text = [text] if isinstance(text, str) else text
    model_type = request.model_type

    # model inference [2 LINES]
    if model_type == 'ml':
        ...
    else:
        ...

    # response
    result = {str(i): {'text': t, 'label': l, 'confidence': c}
                         for i, (t, l, c) in enumerate(zip(text, predictions[0], predictions[1]))}

    return ResponseModel(prediction=result)
```
<br>
## 클라이언트
서빙된 모델을 클라이언트 코드를 활용하여 테스트합니다.

1. Test using request
    ```bash
    import requests

    predict_url = 'http://127.0.0.1:8000/predict'
    data = {"text": "정말 재밌게 잘봤습니다!"}

    predict_response = requests.post(predict_url, data=json.dumps(data))
    print(predict_response.json())
    ```
2. Test using Swagger UI (only available on FastAPI)
    - access to `http://127.0.0.1:8000/docs` using browser
