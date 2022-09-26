import json
import requests

# FastAPI
# root
root_url = 'http://127.0.0.1:8000/'
response_root = requests.get(root_url)
print(response_root.json())

# predict
predict_url = 'http://127.0.0.1:8000/predict'
data = {"text": "정말 재밌게 잘봤습니다!"}

predict_response = requests.post(predict_url, data=json.dumps(data))
print(predict_response.json())


# Flask
# root
root_url = 'http://127.0.0.1:5000/'
response_root = requests.get(root_url)
print(response_root.json())

# predict
predict_url = 'http://127.0.0.1:5000/predict'
data = {"text": "정말 재밌게 잘봤습니다!"}

predict_response = requests.post(predict_url, data=json.dumps(data))
print(predict_response.json())

