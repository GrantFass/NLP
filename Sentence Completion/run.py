from flask import Flask, request, jsonify
import structlog  # for event logging
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
from tqdm import tqdm
import nltk
# from nltk.util import ngrams
import nltk.data
import re
import contractions
from bs4 import BeautifulSoup
import unidecode
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score
# from sklearn.metrics import f1_score, recall_score
# from sklearn.metrics import classification_report
# import math
import json
from pathlib import Path
# import subprocess
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams

nltk.download([
"names",
"stopwords",
"state_union",
"twitter_samples",
"movie_reviews",
"averaged_perceptron_tagger",
"vader_lexicon",
"punkt",
])

lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
encoder = LabelEncoder()
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tqdm.pandas()
n = 3


# create the flask app for the rest endpoints
app = Flask(__name__)

# load the environment files
load_dotenv()

# set up the structured logging file for endpoints
with open(os.getenv('ENDPOINT_LOG_PATH'), "wt", encoding="utf-8") as log_fl:
    structlog.configure(
        processors=[structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer()],
        logger_factory=structlog.WriteLoggerFactory(file=log_fl))

def run_flask():
    with open(os.getenv('ENDPOINT_LOG_PATH'), "wt", encoding="utf-8") as log_fl:
        structlog.configure(
            processors=[structlog.processors.TimeStamper(fmt="iso"),
                        structlog.processors.JSONRenderer()],
            logger_factory=structlog.WriteLoggerFactory(file=log_fl))
        app.run(debug=True, port=9000, host='0.0.0.0', use_reloader=False)  # Added host= so I can access from devices on same network
        
def clean_base(x: str):
        # remove any html tags
        x = BeautifulSoup(x, "html.parser").get_text(separator=" ")
        # # set all to lower
        # x = x.lower()
        # clean up the contractions
        x = contractions.fix(x)
        # remove accended characters
        x = unidecode.unidecode(x)
        # # remove stopwords: https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python
        # x = ' '.join([word for word in x.split() if word not in cachedStopWords]) # slower to use word tokenize
        # # fix punctuation spacing
        # x = re.sub(r'(?<=[\.\,\?])(?=[^\s])', r' ', x)
        # # strip punctuation
        # x = re.sub(r'[\.\,\?\\\/\<\>\;\:\[\]\{\}]', r'', x)
        # strip quotes
        # x = x.replace('\'', '').replace('\"', '')
        x = x.replace('\"', '')
        # # remove some actions
        # remove_list = ['(Laughter)', '(laughter)', '(Music)', '(music)', '(Music ends)', '(Audience cheers)', '(Applause)', '(Applause ends)', '(Applause continues)', '(Bells)', '(Trumpet)', '(Clears throat)']
        # x = ' '.join([word for word in x.split() if word not in remove_list])
        # remove extraneous items
        x = x.replace(' -- ', '').replace(' .. ', ' ').replace(' ... ', ' ')
        # remove extra whitespace
        x = ' '.join(x.strip().split())
        # # may want to add lematization
        # x = ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
        # remove some of the extra bracket tags
        x = re.sub(r"\s{2,}", " ", re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", "", x))
        return x

def from_pickle(filename='NLP/Sentence Completion/freq_dist.pkl'):
    model = {}
    # print(subprocess.run("ls", shell=True, check=True))
    with open(Path(filename), 'rb') as fp:
        model = pickle.load(fp)
    return model
        
def predict_2(row, lm,  mask="_____"):
    # tokenize the inbound question
    tokens = word_tokenize(row['question'])
    question = list(pad_both_ends(tokens, n=2))
    # determine where the mask is without knowing how many masks there are.
    # note that .index is O(n) so we may as well iterate through ourselves to be more verbose.
    mask_indices = []
    mask_count = 0
    for idx in range(len(question)):
        if question[idx] == mask:
            # store the index of the mask
            mask_indices.append(idx)
            # convert the mask to one with a number for future reference
            question[idx] = mask + str(mask_count)
            mask_count += 1
    # pull out the windows
    grams = list(everygrams(question, min_len=1, max_len=n))
    windows = []
    for i in range(len(mask_indices)):
        remask = mask + str(i)
        for gram in grams:
            if remask in gram:
                windows.append(gram)
    # extract the possible solutions
    if mask_indices:
        column_names = ['a)', 'b)', 'c)', 'd)', 'e)']
        solution_likelihoods = []
        for name in column_names:
            if name in row and isinstance(row[name], str) and row[name] != "":
                # tokenize the input
                tokens = word_tokenize(row[name])
                tokens = list(pad_both_ends(tokens, n=2))
                # calculate the probability
                log_likelihood = 0
                custom_windows = []
                for window in windows:
                    # # fill in the blanks in the windows with the possible solutions.
                    new_window = window
                    for i in range(len(mask_indices)):
                        remask = mask + str(i)
                        new_window = list(map(lambda x: x.replace(remask, tokens[mask_indices[i]]), new_window))
                    custom_windows.append(new_window)
                    log_likelihood += lm.score(new_window[-1], new_window[0:-1])
                solution_likelihoods.append((name, log_likelihood))
        solution_likelihoods.sort(key = lambda x: x[1], reverse = True)
        ans = solution_likelihoods[0][0]
        return ans.replace(")", "")

@app.route('/metrics', methods=['GET'])
def get_metrics_endpoint():
    global metrics
    return json.dumps(metrics)
        
@app.route('/predict', methods=['GET'])
def prediction_endpoint():
    global lm
    data = json.loads(request.data.decode('utf-8'))
    print(json.dumps(data))
    for header in ['question', 'a)', 'b)', 'c)', 'd)', 'e)']:
        if header not in data:
            return json.dumps({'error':'Missing Header: ' + header})
    df = pd.DataFrame.from_records([data])
    prediction = predict_2(df.iloc[0], lm)
    # question = data['question']
    # a = data['a']
    # b = data['b']
    # c = data['c']
    # d = data['d']
    # e = data['e']
    response = {'prediction': prediction}
    return json.dumps(response)
    
if __name__ == '__main__':
    global lm
    lm = from_pickle(filename=Path('NLP/Sentence Completion/lm.pkl'))
    metrics = from_pickle(filename=Path('NLP/Sentence Completion/metrics.pkl'))
    run_flask()