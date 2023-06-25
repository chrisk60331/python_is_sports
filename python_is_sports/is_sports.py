from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import sys
import json
import spacy
import os
import re

TRAINED_DATA = os.environ.get("TRAININGDATA", 'training_data.json')
SPORTS = 'sports'
VECTORIZER = 'vectorizer.pkl'
MODEL = 'model.pkl'
def logisticregression(data):
    texts, labels = zip(*data)
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    model = LogisticRegression()
    model.fit(X_train, labels_train)
    joblib.dump(model, MODEL)
    joblib.dump(vectorizer, VECTORIZER)
    return model, vectorizer

def labeldata(data, label):
    return [(title[:100], label) for title in data]

def predict_article(model, vectorizer, text):
    X = vectorizer.transform(text)
    prediction = model.predict(X)
    return prediction[0] == SPORTS


def replace_useless(text, nlp):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    doc = nlp(text)
    tokens = [
        token.text
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and token.pos_ in ["NOUN", "VERB", "PROPN", "NUM"]
    ]
    return " ".join(tokens)

def get_data(file_name, label, nlp):
    data = []
    for indx in range(0, 10000, 100):
        data_file = f"{file_name}{indx}.json"
        if not os.path.exists(data_file):
            raise Exception(f"Data file {data_file} not found! Did you forget to run get_data.py?")
        new = labeldata(
            [
                replace_useless(f"{x.get('title')} - {x.get('description')}", nlp)[:150]
                for x in json.loads(open(data_file, 'r').read()).get('data')
            ], label
        )
        data.extend(new)
    return data


def preprocess(nlp):
    business = get_data("business,-sports", "notsports", nlp)
    sports = get_data("sports,-business", "sports", nlp)
    print("sports: ", len(sports), "notsports: ", len(business))
    data = sports + business
    open('training_data.json', 'w').write(json.dumps(data))


def run_prediction(text):
    nlp = spacy.load("en_core_web_sm")
    if not os.path.exists(MODEL) or not os.path.exists(VECTORIZER):
        if not os.path.exists(TRAINED_DATA):
            preprocess(nlp)
        data =  json.loads(open(TRAINED_DATA).read())
        model, vectorizer = logisticregression(data)
    else:
        model = joblib.load(MODEL)
        vectorizer = joblib.load(VECTORIZER)
    try:
        text = [replace_useless((text), nlp)]
        return predict_article(model, vectorizer, text)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    print(run_prediction(" ".join(sys.argv[1:])))

