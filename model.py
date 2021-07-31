from time import time
import re

import nltk
import numpy as np
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import wordnet
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

#         X, y = self._validate_data(X, y)

        # Model performance should be decoupled from references to training data
        self._Vectors = np.copy(X)
        self._labels = np.copy(y)

        return self

    def _gram_matrices(self, X):
        try:
            print(self._Vectors.shape)
            gram_matrix = linear_kernel(X, self._Vectors)
        except MemoryError:
            print(X.shape)
            print(self._Vectors.shape)
            raise MemoryError(f'too big {X.shape}, {self._Vectors.shape}')
        gram_desc_args = np.fliplr(gram_matrix.argsort())
        gram_desc = np.take_along_axis(gram_matrix, gram_desc_args, axis=1)

        return gram_matrix, gram_desc_args, gram_desc

    def predict(self, X, show_score=True, quiet=False):
        start = time()
        if type(X) == str:
            X = [X]
            
        print(self._Vectors.shape)

        gram_matrix, gram_desc_args, gram_desc = self._gram_matrices(X)
        pred = self._labels.take(gram_desc_args[:, :self.n_best], axis=0)
        score = gram_desc[:, :self.n_best]
        
        # Simplify output for single inference
#         if X.shape[0] == 1:
#             pred = pred[0]
#             score = score[0]

        None if quiet else print(pred)
        if show_score:
            None if quiet else print(score)
        None if quiet else print('Prediction took', time() - start, 'seconds')

        return pred, score


class LemmaTokenizer:

    def __init__(self, custom=False):
        self.wnl = WordNetLemmatizer()
        self.custom = custom

    def __call__(self, doc):
        if self.custom:
            # Find alphabetical tokens at least 3 chars long
            tokens = re.findall(r"[a-zA-Z]{3,}", doc)
            tokens = [word for word in tokens if len(word) >= 3]

            # Only use verb/noun tokens
            token_tags = nltk.pos_tag(tokens)
            tokens = []
            
            # Convert NLTK POS tags to WordNet POS tags
            for token, tag in token_tags:
                if tag[0] == 'V':
                    tokens.append((token, wordnet.VERB))
                elif tag[0] == 'N':
                    tokens.append((token, wordnet.NOUN))

        else:
            tokens = [(t,) for t in word_tokenize(doc)]

        lemmatized_tokens = [self.wnl.lemmatize(*t) for t in tokens]
        return lemmatized_tokens


def get_fitted_model(corpus, labels, lemmatize='default', **hyperparams):
    print('Training model...')

    start = time()
    pipe = get_model(lemmatize, **hyperparams)
    pipe.fit(corpus, labels)
    print(len(pipe[0].get_feature_names()))
    print('Took', time() - start, 'seconds')
    return pipe


def get_model(lemmatize, **hyperparams):
    # Set lemmatization, if any
    lemmatize = lemmatize.lower()
    if lemmatize == 'default':
        vectorizer = TfidfVectorizer(
            tokenizer=LemmaTokenizer(),
            stop_words='english'
        )
    elif lemmatize == 'custom':
        vectorizer = TfidfVectorizer(
            tokenizer=LemmaTokenizer(custom=True),
            stop_words='english'
        )
    elif lemmatize == 'none':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        raise ValueError('lemmatize must be {default, custom, none}')

    # Create and train pipeline
    pipe = make_pipeline(
        vectorizer,
        VectorSimilarity(**hyperparams)
    )

    return pipe
