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
    def __init__(self, n_best=10):
        self.n_best = n_best
        self._Vectors = np.array([])
        self._labels = np.array([])

    def fit(self, X, y):
        # Convert dense (i.e. "efficient") array representation to sparse
        if not isinstance(X, (np.ndarray, np.generic)):
            X = X.toarray()

        # Handle case where X/y are basic lists
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have first axes of equal size.')

        # Model performance should be decoupled from references to training data
        self._Vectors = np.copy(X)
        self._labels = np.copy(y)

        return self

    def _gram_matrices(self, X):
        try:
            # print(self._Vectors.shape)
            gram_matrix = linear_kernel(X, self._Vectors)
        except MemoryError:
            # print(X.shape)
            # print(self._Vectors.shape)
            raise MemoryError(f'too big {X.shape}, {self._Vectors.shape}')
        gram_desc_args = np.fliplr(gram_matrix.argsort())
        gram_desc = np.take_along_axis(gram_matrix, gram_desc_args, axis=1)

        return gram_matrix, gram_desc_args, gram_desc

    def predict(self, X, verbose=False):
        gram_matrix, gram_desc_args, gram_desc = self._gram_matrices(X)
        pred = self._labels.take(gram_desc_args[:, :self.n_best], axis=0)
        score = gram_desc[:, :self.n_best]
        
        # TODO: Simplify output for single inference
#         if X.shape[0] == 1:
#             pred = pred[0]
#             score = score[0]

        return pred, score


class LemmaTokenizer:
    def __init__(self, custom=False):
        self._wnl = WordNetLemmatizer()
        self.custom = custom # Whether to use custom tokenization pipeline

    def __call__(self, doc):
        if self.custom:
            # Find alphabetical tokens at least 3 chars long
            tokens = re.findall(r"[a-zA-Z]{3,}", doc)
            tokens = [tok for tok in tokens if len(tok) >= 3]

            # Get NLTK POS tags
            token_tags_nltk = nltk.pos_tag(tokens)
            token_tags = []

            # Only use nouns and verbs, convert NLTK POS tags to WordNet POS tags
            for token, tag in token_tags_nltk:
                if tag[0] == 'V':
                    token_tags.append((token, wordnet.VERB))
                elif tag[0] == 'N':
                    token_tags.append((token, wordnet.NOUN))

        else:
            # Add "null" tags for auto-lemmatization
            token_tags = [(t,) for t in word_tokenize(doc)]

        lemmatized_tokens = [self._wnl.lemmatize(*t) for t in token_tags]
        lemmatized_tokens = [tok for tok in lemmatized_tokens if len(tok) >= 3]
        return lemmatized_tokens


def get_vectorizer(lemmatize='default'):
    lemmatize = lemmatize.lower()
    vectorizer = TfidfVectorizer(stop_words='english')

    # Ugly dict unpacking to pass typecheck lol
    if lemmatize == 'default':
        vectorizer.set_params(**{'tokenizer': LemmaTokenizer()})
    elif lemmatize == 'custom':
        vectorizer.set_params(**{'tokenizer': LemmaTokenizer(custom=True)})
    elif lemmatize == 'none':
        pass
    else:
        raise ValueError('lemmatize must be {default, custom, none}')

    return vectorizer


class TfidfPredictor(BaseEstimator):
    def __init__(self, lemmatize='default', n_best=10):
        self._vectorizer = get_vectorizer(lemmatize=lemmatize)
        self._similarity = VectorSimilarity(n_best=n_best)
        self._pipe = make_pipeline(
            self._vectorizer,
            self._similarity
        )

    def fit(self, corpus, labels, verbose=False):
        start = time()
        self._pipe.fit(corpus, labels)
        print('Training took', time() - start, 'seconds') if verbose else None

    def predict(self, X, verbose=False):
        start = time()
        if type(X) == str:
            X = [X]
        pred, score = self._pipe.predict(X)
        print('Prediction took', time() - start, 'seconds') if verbose else None
        return pred, score

    def inspect_doc(self, doc, n_top=10):
        if type(doc) == str:
            doc = [doc]
        if len(doc) > 1:
            raise ValueError('Only one document per call is supported')

        vocab = np.array(self._vectorizer.get_feature_names(), ndmin=2)
        weights = self._vectorizer.transform(doc).toarray()

        weights_desc_args = np.flip(weights.argsort())
        words_desc = np.take_along_axis(vocab, weights_desc_args, axis=1)
        weights_desc = np.take_along_axis(weights, weights_desc_args, axis=1)

        top_words = words_desc[:, :n_top]
        top_weights = weights_desc[:, :n_top]

        return list(zip(top_words, top_weights))

    def get_weights(self, query, doc):
        if type(doc) == str:
            doc = [doc]

        tokenizer = self._vectorizer.build_tokenizer()
        tokens = tokenizer(query)
        vocab = np.array(self._vectorizer.get_feature_names(), ndmin=2)

        indices = []
        for tok in tokens:
            indices.append(np.argwhere(vocab == tok)[0, 1])
        indices = np.array(indices)

        weights = self._vectorizer.transform(doc).toarray().flatten()
        weights_desc = np.take_along_axis(weights, indices, axis=0)[:len(tokens)]

        return list(zip(tokens, weights_desc))
