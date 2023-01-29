from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template
# from flask_debugtoolbar import DebugToolbarExtension
import pickle
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import nltk
# Download dependency
for dependency in (
    "omw-1.4",
    "stopwords",
    "wordnet",
    "punkt",
    "averaged_perceptron_tagger"
):
    nltk.download(dependency)

app = Flask(__name__)  # creation application
app.debug = True
# toolbar = DebugToolbarExtension(app)
app.static_folder = 'static'

# LOAD MODEL

MODEL_VERSION = 'lstm_model_rus.h5'  # modèle
MODEL_PATH = os.path.join(os.getcwd(), 'models',
                          MODEL_VERSION)  # path vers le modèle
model = load_model(MODEL_PATH)  # chargement du modèle

# LOAD TOKENIZER

TOKENIZER_VERSION = 'tokenizer_rus.pickle'
TOKENIZER_PATH = os.path.join(os.getcwd(), 'models',
                              TOKENIZER_VERSION)  # path vers le tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)


if __name__ == '__main__':  # faire run l'application
    app.run(debug=True, use_debugger=True)