from time import time
import re

import nltk
import numpy as np
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline

nltk.download([
    'punkt',
    'wordnet',
    'tagsets',
    'averaged_perceptron_tagger'
    ], quiet=True
)


class VectorSimilarity(BaseEstimator):
    _Vectors = None
    _labels = None

    def __init__(self, n_best=10):
        self.n_best = n_best

    def fit(self, X, y):
        # Required to pass check_estimator()
        if X.dtype == np.dtype('complex128'):
            raise ValueError('Complex data not supported')

        # Convert dense (i.e. "efficient") array representation to sparse
        if not isinstance(X, (np.ndarray, np.generic)):
            X = X.toarray()

        X, y = self._validate_data(X, y)

        # Model performance should be decoupled from references to training data
        self._Vectors = np.copy(X)
        self._labels = np.copy(y)

        return self

    def _gram_matrices(self, X):
        gram_matrix = linear_kernel(X, self._Vectors)
        gram_desc_args = np.fliplr(gram_matrix.argsort())
        gram_desc = np.take_along_axis(gram_matrix, gram_desc_args, axis=1)

        return gram_matrix, gram_desc_args, gram_desc

    def predict(self, X):
        return self.predict_score(X)[0]

    def score(self, X, y=None):
        return self.predict_score(X)[1]

    def predict_score(self, X):
        # Ensures that any call to predict/score will require only one call to linear_kernel()
        gram_matrix, gram_desc_args, gram_desc = self._gram_matrices(X)

        pred = self._labels.take(gram_desc_args[:, :self.n_best])
        score = gram_desc[:, :self.n_best]

        return pred, score


class LemmaTokenizer:
    def __init__(self, custom=False):
        self.wnl = WordNetLemmatizer()
        self.custom = custom

    def __call__(self, doc):
        if self.custom:
            # Find alphabetical tokens at least 3 chars long
            tokens = re.findall(r"(?u)\b\w\w+\b", doc)
            tokens = [word for word in tokens if len(word) >= 3]

            # Only use verb/noun tokens
            tags = nltk.pos_tag(tokens)
            tokens = [word for word, tag in tags if tag[0] in ['V', 'N']]

        else:
            tokens = word_tokenize(doc)

        lemmatized_tokens = [self.wnl.lemmatize(t) for t in tokens]
        return lemmatized_tokens


def get_fitted_model(corpus, labels, lemmatize='default', **hyperparams):
    print('Training model...')
    start = time()
    pipe = get_model(lemmatize, **hyperparams)

    pipe.fit(corpus, labels)
    print('Took', time() - start, 'seconds')

    return pipe


def get_model(lemmatize, **hyperparams):
    # Set lemmatization, if any
    lemmatize = lemmatize.lower()
    if lemmatize == 'default':
        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())
    elif lemmatize == 'custom':
        vectorizer = TfidfVectorizer(
            tokenizer=LemmaTokenizer(custom=True)
        )
    elif lemmatize == 'none':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError('lemmatize must be {default, custom, none}')

    # Create and train pipeline
    pipe = make_pipeline(
        vectorizer,
        VectorSimilarity(n_best=hyperparams['n_best'])
    )

    # Add the predict_score() function from VectorSimilarity - inelegant, but gets the job done
    pipe.predict_score = lambda x: pipe[1].predict_score(pipe[0].transform(x))
    return pipe


def infer(pipe, text, show_score=False):
    print('Inferring on the query:', text)
    start = time()
    if type(text) == str:
        text = list(text)

    print(pipe.predict(text))

    if show_score:
        print(pipe.score(text))
    print('Took', time() - start, 'seconds')
