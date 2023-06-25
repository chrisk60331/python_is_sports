from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import sys
import json
import spacy
import os
import re


def logisticregression(data):
    texts, labels = zip(*data)
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    model = LogisticRegression()
    model.fit(X_train, labels_train)
    predictions = model.predict(X_test)

    return model, vectorizer

def prepdata(data):
    parsed = []
    for title, desc in data:
        soup = BeautifulSoup(desc, 'html.parser')
        parsed.append(f"{title}: {soup.get_text()}")
    return parsed

def labeldata(data, label):
    return [(title[:100], label) for title in data]

def parserss(file_name):
    feed = ET.parse(file_name).getroot()
    channels =  feed.findall('channel')
    return [
        (item.find("title").text,item.find("description").text)
        for channel in channels
        for item in channel.findall('item')
    ]

def predict_article(model, vectorizer, text):
    X = vectorizer.transform(text)
    prediction = model.predict(X)
    return prediction


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
    for indx in range(0, 1000, 100):
        new = labeldata(
            [
                replace_useless(f"{x.get('title')} - {x.get('description')}", nlp)
                for x in json.loads(open(f"{file_name}{indx}.json", 'r').read()).get('data')
            ], label
        )
        data.extend(new)
    return data


def preprocess():
    nlp = spacy.load("en_core_web_sm")
    business = get_data("business,-sports", "notsports", nlp)
    sports = get_data("sports,-business", "sports", nlp)
    print("sports: ", len(sports), "notsports: ", len(business))
    data = sports + business
    open('training_data.json', 'w').write(json.dumps(data))

if __name__ == '__main__':

    if not os.path.exists('training_data.json'):
        preprocess()
    try:
        nlp = spacy.load("en_core_web_sm")
        data =  json.loads(open('training_data.json').read())
        model, vectorizer = logisticregression(data)
        text = [replace_useless(" ".join(sys.argv[1:]), nlp)]
        print(text)
        print(predict_article(model, vectorizer, text)[0])
    except Exception as e:
        print(e)

