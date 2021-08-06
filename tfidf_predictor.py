from time import time
from collections import Sequence
import re

import nltk
import numpy as np
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import wordnet
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from numba import jit, prange

nltk.download([
    'punkt',
    'wordnet',
    'tagsets',
    'averaged_perceptron_tagger'
], quiet=True
)

# Not working at the moment - types issue
@jit(nopython=True)
def linear_kernel_numba(u:np.ndarray, M:np.ndarray):
    scores = np.zeros(M.shape[0])
    for i in prange(M.shape[0]):
        v = M[i]
        m = u.shape[0]
        udotv = 0
        u_norm = 0
        v_norm = 0
        for j in range(m):
            if (np.isnan(u[j])) or (np.isnan(v[j])):
                continue

            udotv += u[j] * v[j]
            u_norm += u[j] * u[j]
            v_norm += v[j] * v[j]

        u_norm = np.sqrt(u_norm)
        v_norm = np.sqrt(v_norm)

        if (u_norm == 0) or (v_norm == 0):
            ratio = 1.0
        else:
            ratio = udotv / (u_norm * v_norm)
        scores[i] = ratio
    return scores

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
            gram_matrix = linear_kernel(X, self._Vectors)
#             X = np.array(X, dtype='float64')
#             gram_matrix = linear_kernel_numba(X, self._Vectors)
        except MemoryError:
            raise MemoryError(f'too big {X.shape}, {self._Vectors.shape}')
        gram_desc_args = np.fliplr(gram_matrix.argsort())
        gram_desc = np.take_along_axis(gram_matrix, gram_desc_args, axis=1)

        return gram_matrix, gram_desc_args, gram_desc

    def predict(self, X, verbose=False):
        gram_matrix, gram_desc_args, gram_desc = self._gram_matrices(X)
        pred = self._labels.take(gram_desc_args[:, :self.n_best], axis=0)
        score = gram_desc[:, :self.n_best]

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
    def __init__(self, 
                 lemmatize='default', 
                 n_best=10, 
                 label_names=None):
        
        self._vectorizer = get_vectorizer(lemmatize=lemmatize)
        self._similarity = VectorSimilarity(n_best=n_best)
        self._pipe = make_pipeline(
            self._vectorizer,
            self._similarity
        )
        self.label_names = label_names

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
    
    def predict_obj(self, X, verbose=False):
        if type(X) != str:
            raise ValueError('predict_obj only supports one inference at a time.')
            
        preds, scores = self.predict(X, verbose=verbose)
        
        # Get length of each prediction entry
        if len(self._similarity._labels.shape) < 2:
            num_labels = 1
        else:
            num_labels = self._similarity._labels.shape[-1]
        
        if self.label_names == None or not isinstance(self.label_names, Sequence):
            raise ValueError('predict_obj requires a sequence of label names for the output.')
        if len(self.label_names) != num_labels:
            raise ValueError('label_names does not have the same size as the labels configured on this predictor.')
        
        
        output = {
            'Text': X,
            'Similar': []
        }

        for pred, score in zip(preds[0], scores[0]):
            label_val = zip(self.label_names, pred)
            entry = {label: val for label, val in label_val}
            entry['Score'] = score
            output['Similar'].append(entry)
            
        return output
            

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

        return list(zip(top_words[0], top_weights[0]))

    def get_weights(self, query, doc):
        if type(doc) == str:
            doc = [doc]
        if len(doc) > 1 or type(query) != str:
            raise ValueError('Only one document per call is supported')

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
