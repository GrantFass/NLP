from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import random
from tqdm.notebook import tqdm
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