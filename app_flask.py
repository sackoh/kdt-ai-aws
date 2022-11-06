from flask import Flask, request, json
from model import MLModelHandler, DLModelHandler

app = Flask(__name__)

# assign model handler as global variable [2 LINES]


@app.route("/predict", methods=["POST"])
def predict():
    # handle request and body
    body = request.get_json()
    text = body.get('text', '')
    text = [text] if isinstance(text, str) else text
    model_type = body.get('model_type', 'ml')
    use_gpu = body.get('use_gpu', False)

    # model inference [2 LINES]
    if model_type == 'ml:
        pass
    else:
        pass

    # response
    result = json.dumps({str(i): {'text': t, 'label': l, 'confidence': c}
                         for i, (t, l, c) in enumerate(zip(text, predictions[0], predictions[1]))})
    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)
