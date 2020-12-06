import os
import logging
import requests
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from utils import clean_text

logger = logging.getLogger(__name__)


def download_data(mode):
    base_url = f'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_{mode}.txt'
    r = requests.get(base_url)
    with open(f'ratings_{mode}.txt', 'wb') as w:
        w.write(r.content)
    logger.info("Downloaded from {}".format(base_url))


def train_and_evaluate():
    train = pd.read_csv('ratings_train.txt', sep='\t').drop('id', axis=1).fillna('')
    test = pd.read_csv('ratings_test.txt', sep='\t').drop('id', axis=1).fillna('')
    X_train, y_train = train['document'].apply(clean_text), train['label']
    X_test, y_test = test['document'].apply(clean_text), test['label']

    vectorizer = CountVectorizer(max_features=100000)
    vectorizer.fit(X_train)
    logger.info("fitting Counter vectorizer")

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    logger.info("Transform raw text into vector")

    model = MultinomialNB()
    model.fit(X_train, y_train)
    logger.info("Trained Naive Bayes model.")
    evaluation_score = model.score(X_test, y_test)
    logger.info(f"ML model accuracy score: {evaluation_score*100:.2f}%")

    return model, vectorizer


def serialization(model, vectorizer):
    import joblib
    os.makedirs('model', exist_ok=True)
    joblib.dump(vectorizer, 'model/ml_vectorizer.pkl')
    logger.info(f'Saved vectorizer to `model/ml_vectorizer.pkl`')
    joblib.dump(model, 'model/ml_model.pkl')
    logger.info(f'Saved model to `model/ml_model.pkl`')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    start_time = datetime.now()
    # Download train and test data from github
    for mode in ['train', 'test']:
        download_data(mode)

    # train and evaluate model
    model, vectorizer = train_and_evaluate()

    # Serialization
    serialization(model, vectorizer)
    logger.info(f"Elapsed time : {datetime.now() - start_time}")
