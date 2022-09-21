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
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self, ):
        # De-serializing model and loading vectorizer
        
        pass

    def preprocess(self, data):
        # cleansing raw text

        # vectorizing cleaned text

        return data

    def inference(self, data):
        # get predictions from model as probabilities
        
        return data

    def postprocess(self, data):
        # process predictions to predicted label and output format

        return data

    def handle(self, data):
        # do above processes

        return data


class DLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self, ):
        pass

    def preprocess(self, data):
        return data

    def inference(self, data):
        return data

    def postprocess(self, data):
        return data

    def handle(self, data):
        return data
