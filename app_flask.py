from flask import Flask, request, json
from model import MLModelHandler, get_ML_Handler

app = Flask(__name__)

# assign model handler as global variable
classifier = get_ML_Handler()

@app.route("/")
def root():
    return {"message": "Hello World"}


@app.route("/predict", methods=["POST"])
def predict():
    # handle request and body
    body = request.get_json(force=True)
    text = body.get('text', '')

    prediction = classifier.handle(text)
    result = json.dumps({'prediction':prediction})

    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)
