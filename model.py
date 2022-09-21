import joblib
import itertools
from utils import clean_text

class ModelHandler:
    def __init__(self):
        self.id2label = {0: 'negative', 1: 'positive'}

    def _clean_text(self, text):
        model_input = []
        if isinstance(text, str):
            cleaned_text = clean_text(text)
            model_input.append(cleaned_text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and (all(isinstance(t, str) for t in text)):
            cleaned_text = itertools.chain((clean_text(t) for t in text))
            model_input.extend(cleaned_text)
        else:
            model_input.append('')
        return model_input


class MLModelHandler(ModelHandler):
    def __init__(self, vectorizer_path, model_path):
        super().__init__()
        self.initialize(vectorizer_path, model_path)

    def initialize(self, vectorizer_path, model_path):
        # De-serializing model and loading vectorizer
        self.vectorizer = joblib.load(vectorizer_path) 
        self.model = joblib.load(model_path) 

    def preprocess(self, data):
        # cleansing raw text        
        cleaned_data = self._clean_text(data)

        # vectorizing cleaned text
        vectorized_data = self.vectorizer.transform(cleaned_data)
        return vectorized_data

    def inference(self, data):
        # get predictions from model as probabilities
        model_output = self.model.predict_proba(data)[0]

        return model_output

    def postprocess(self, data):
        # process predictions to predicted label and output format
        
        predict_index = data.argmax()
        print(predict_index)
        if predict_index == 1:
            prediction = 'positive'
        else:
            prediction = 'negative'
        score = data[predict_index]
        result = {'label': prediction, 'score':score}
        return result

    def handle(self, data):
        # do above processes
        data = self.preprocess(data)
        model_output = self.inference(data)
        result = self.postprocess(model_output)
        return result


class DLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):
        pass

    def preprocess(self, data):
        return data

    def inference(self, data):
        return data

    def postprocess(self, data):
        return data

    def handle(self, data):
        return data


def get_ML_Handler():
    ml_handler = MLModelHandler(vectorizer_path='model/ml_vectorizer.pkl',
                                model_path='model/ml_model.pkl')
    return ml_handler


def get_DL_Handler():
    return DLModelHandler()


