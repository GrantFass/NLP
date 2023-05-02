from flask import Flask, request, jsonify
import structlog  # for event logging
from dotenv import load_dotenv
import os
import pickle

from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import random
from tqdm import tqdm
import nltk
from nltk.util import ngrams
import nltk.data
import re
import contractions
from bs4 import BeautifulSoup
import unidecode
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import json

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
        x = x.replace('\'', '').replace('\"', '')
        # remove some actions
        remove_list = ['(Laughter)', '(laughter)', '(Music)', '(music)', '(Music ends)', '(Audience cheers)', '(Applause)', '(Applause ends)', '(Applause continues)', '(Bells)', '(Trumpet)', '(Clears throat)']
        x = ' '.join([word for word in x.split() if word not in remove_list])
        # remove extraneous items
        x = x.replace(' -- ', '').replace(' .. ', ' ').replace(' ... ', ' ')
        # remove extra whitespace
        x = ' '.join(x.strip().split())
        # # may want to add lematization
        # x = ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
        # remove some of the extra bracket tags
        x = re.sub(r"\s{2,}", " ", re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", "", x))
        return x
        
def tokenize_question(question):
    if not isinstance(question, str) or question is None:
        return []
    clean = clean_base(question)
    sent = word_tokenize(clean)
    start_of_sentence = "<sent>"
    end_of_sentence = "<\\sent>"
    sent.insert(0, start_of_sentence)
    sent.append(end_of_sentence)
    return sent

def get_mask_indices(row, mask='_____'):
    """Returns a list containing the integer indices of where the mask occurs for a given row of the SAT dataset.

    Args:
        row : a given row of the SAT dataset
        mask (str, optional): the mask to look for the indicies of occurance of. Defaults to '_____'.

    Returns:
        list: contains integer indices of where the mask occurs for a given row of the SAT dataset
    """
    mask_indices = []
    previous_idx = 0
    for blank_id in range(row['blanks']):
        idx = tokenize_question(row['question']).index(mask, previous_idx)
        mask_indices.append(idx)
        previous_idx = idx + 1
    return mask_indices

def extract_possible_solutions(row, mask_indices):
    """computes the possible solutions that can fill in the blanks for a given question in a row of the SAT dataset.
    The possible solutions come from the a-e columns of the row while the question itself comes from the question column.

    Args:
        row (_type_): a given row of the SAT dataset.
        mask_indices (list): list containing integer indices of where the mask occurs for a given row of the SAT dataset.

    Returns:
        list: a list of tuples. Each tuple contains the possible solutions in their respective order to the question.
    """
    possible_solutions = []
    for i in ['a)', 'b)', 'c)', 'd)', 'e)']:
        if i in row and row[i]:
            # print(test[i])
            tokens = tokenize_question(row[i])
            if tokens:
                possible_solution = []
                for mask_idx in mask_indices:
                    possible_solution.append(tokens[mask_idx])
                possible_solutions.append(tuple(possible_solution))
    return possible_solutions

def get_ngrams_with_mask(mask_indices, tokenized_question, n, mask='_____'):
    """This method is used to compute the sliding window for the original n-gram size as well
    as each of the subsequently smaller sizes of n. This method requires the variable 'n' to
    be defined globally for the maximum n-gram size to look for.

    Args:
        mask_indices (list): list containing integer indices of where the mask occurs for a given row of the SAT dataset.
        tokenized_question (list): list comprised of the individual tokens that were parsed from the input question
        n (int): the maximum n-gram size to use.
        mask (str, optional): the mask to look for the indicies of occurance of. Defaults to '_____'.
        

    Returns:
        list: A list containing tuples of each n-gram is returned. If more than one 
    """
    ranges = []
    count = 0
    for mask_idx in mask_indices:
        remask = mask + str(count)
        tokenized_question[mask_idx] = remask
        count += 1
        # print("\nmask_idx = %d" % mask_idx)
        for i in range(n, 0, -1):  # for each successively smaller n-gram size
            # print("n = %d" % i)
            upper_bound = mask_idx + 1
            lower_bound = mask_idx - i + 1
            if lower_bound >= 0 and upper_bound < len(tokenized_question):
                ranges.append((lower_bound, upper_bound))
                # print("range(%d, %d)" % (lower_bound, upper_bound))
            for j in range(1, i):  # already processed first n-gram for size i. Now process the rest where j is the offset.
                upper_bound += 1
                lower_bound += 1
                if lower_bound >= 0 and upper_bound < len(tokenized_question):
                    ranges.append((lower_bound, upper_bound))
                    # print("range(%d, %d)" % (lower_bound, upper_bound))
    n_grams = []
    for indices in ranges:
        n_gram = []
        for i in range(indices[0], indices[1]):
            n_gram.append(tokenized_question[i])
        n_grams.append(n_gram)
        
    return n_grams

def find_best_answer(possible_solutions, windows, freq_dist, mask='_____'):
    """method used to determine which one of the possible solutions is the most likely answer.

    Args:
        possible_solutions (list): list of tuples where each tuple is one of the possible solutions to fill in the blanks in the sentence.
        windows (list): the windows of n-grams. These windows are centered around the locations of the blanks in the sentence.
        freq_dist (dict): a dictionary where the key is a n-gram as a tuple and the value is the frequency count of the n-gram tuple.
        mask (str, optional): the mask to look for the indicies of occurance of. Defaults to '_____'.

    Returns:
        (tuple): the tuple containing the words from the best possible solution predicted.
    """
    log_likelihood_given_solution = []
    for possible_solution in possible_solutions:  # compute the log likelihood of each possible solution.
        # print("\nGenerating Solutions For: %s" % (possible_solution,))
        log_likelihood = 0
        for window in windows:  # go through each of the generated windows and find its probability.
            for i in range(len(possible_solution)):  # replace the masks with each of the possible solution words
                window = list(map(lambda x: x.replace(mask + str(i), possible_solution[i]), window))
            # print(window)
            ngram_size = len(window)
            raw_ngram_count = freq_dist[tuple(window)]
            if raw_ngram_count > 0:  # the ngram occurs in our corpus
                ngram_count_of_prior = 1  # default to 1 for unigrams
                if ngram_size > 1:  # not a unigram so we need to find the count of the 'given' ngram
                    prior = window[0: len(window)-1]
                    ngram_count_of_prior = freq_dist[tuple(prior)]
                vocab = 0
                raw_probability = raw_ngram_count / (ngram_count_of_prior + vocab)
                log_probability = math.log10(raw_probability)
                log_likelihood += log_probability
        # print("Log Likelihood = %.2f" % log_likelihood)
        log_likelihood_given_solution.append((possible_solution, log_likelihood))
    log_likelihood_given_solution.sort(key = lambda x: x[1], reverse = True)
    answer = log_likelihood_given_solution[0][0]
    # print("Best Answer: %s" % (answer, ))
    # print("sorted likelihoods per possible solution:\n%s" % str(log_likelihood_given_solution))
    return answer

def determine_prediction_based_on_best_answer(best_answer, row, mask_indices):
    """method to determine the column name of the best answer

    Args:
        best_answer (tuple): the tuple containing the words from the best possible solution predicted.
        row : a row in the pandas data frame for the SAT dataset.
        mask_indices (list): list containing the indices of where the mask (blanks) occur.

    Returns:
        str: the column name corresponding to the best answer.
    """
    for col in ['a)', 'b)', 'c)', 'd)', 'e)']:
        if col in row:
            tokens = tokenize_question(row[col])
            result = True
            count = 0
            for mask_idx in mask_indices:
                if tokens[mask_idx] != best_answer[count]:
                    result = False
                count += 1
            if result:
                return col.replace(")", "")

def from_pickle(filename='Sentence Completion\\freq_dist.pkl'):
    model = {}
    with open(filename, 'rb') as fp:
        model = pickle.load(fp)
    return model

def to_pickle(filename, model):
    with open(filename, 'wb') as fp:
        pickle.dump(model, fp)
        print("Saved %s to file %s" % ('freq_dist', filename))

def count_num_blanks(question, mask="___"):
    question_tokens = tokenize_question(question)
    # question = question.replace(mask, "_____")
    num_blanks = 0
    for token in question_tokens:
        if token == mask:
            num_blanks += 1
    return num_blanks
            
def predict(row, model_filepath='Sentence Completion\\freq_dist.pkl'):
    # load the model
    model = from_pickle(filename=model_filepath)
    # parse the incoming question
    mask = "_____"
    if 'blanks' not in row:
        row['blanks'] = count_num_blanks(row['question'], mask=mask)
    mask_indices = get_mask_indices(row, mask=mask)
    tokenized_question = tokenize_question(row['question'])
    possible_solutions = extract_possible_solutions(row, mask_indices)
    windows = get_ngrams_with_mask(mask_indices, tokenized_question, n=3, mask=mask)
    # Look against the model. Changing the freq_dist used should allow us to try different models.
    # TODO: what if we try to use an ensemble of models with different maximum n-gram sizes?
    best_answer = find_best_answer(possible_solutions, windows, model, mask=mask)  # note that freq_dist is a global here.
    prediction = determine_prediction_based_on_best_answer(best_answer, row, mask_indices)
    return prediction

@app.route('/metrics', methods=['GET'])
def get_metrics_endpoint():
    metrics = from_pickle(filename='Sentence Completion\\metrics.pkl')
    return json.dumps(metrics)
        
@app.route('/predict', methods=['GET'])
def prediction_endpoint():
    data = json.loads(request.data.decode('utf-8'))
    print(json.dumps(data))
    for header in ['question', 'a)', 'b)', 'c)', 'd)', 'e)']:
        if header not in data:
            return json.dumps({'error':'Missing Header: ' + header})
    df = pd.DataFrame.from_records([data])
    prediction = predict(df.iloc[0])
    # question = data['question']
    # a = data['a']
    # b = data['b']
    # c = data['c']
    # d = data['d']
    # e = data['e']
    response = {'prediction': prediction}
    return json.dumps(response)
    
if __name__ == '__main__':
    run_flask()