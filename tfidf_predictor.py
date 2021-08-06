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

# Download NLTK extensions for Lemmatizer and WordNet POS tagger
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
        """Instantiate a VectorSimilarity estimator. VectorSimilarity can perform
        pairwise comparisons of vectors to find the most similar ones, returning
        the corresponding "labels".

        Args:
            n_best (int, optional): How many of the best vectors to return.
            Defaults to 10.
        """        
        self.n_best = n_best
        self._Vectors = np.array([])
        self._labels = np.array([])

    def fit(self, X, y):
        """Fit VectorSimilarity onto a set of vectors and their associated labels.

        Args:
            X (Iterable[Iterable[float]]): List of vectors to be compared
            y (Iterable[any]): Associated labels for each vector. These labels
            themselves can be of any type.

        Raises:
            ValueError: if X and y have different lengths. Each vector can only
            be associated with one label.

        Returns:
            VectorSimilarity: Fitted VectorSimilarity object. Note that this
            function is also an in-place operation.
        """
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
        """Get the Gram matrices (dot product score) and highest-scoring indices
        for a list of input vectors.

        Args:
            X (Iterable[Iterable[float]]): list of input vectors to be compared

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): unsorted Gram matrix (dot
            product), n highest-scoring indices, and n highest scores.
        """        
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
        """Get the labels of the most similar vectors in the training set.

        Args:
            X (Iterable[Iterable[float]]): list of input vectors to be compared
            verbose (bool, optional): whether to print extra information about
            the prediction. Defaults to False.

        Returns:
            (np.ndarray, np.ndarray): the n best labels and dot product scores
            from the training set.
        """
        gram_matrix, gram_desc_args, gram_desc = self._gram_matrices(X)
        pred = self._labels.take(gram_desc_args[:, :self.n_best], axis=0)
        score = gram_desc[:, :self.n_best]

        return pred, score

class LemmaTokenizer:
    def __init__(self, custom=False):
        """Instantiate a LemmaTokenizer. LemmaTokenizer tokenizes text, then
        performs lemmatization (case and tense are removed).

        Args:
            custom (bool, optional): whether to use a custom tokenization
            and lemmatization pipeline. Defaults to False, in which case the
            default NLTK tokenizer is used.
        """        
        self._wnl = WordNetLemmatizer()
        self.custom = custom # Whether to use custom tokenization pipeline

    def __call__(self, doc):
        """Tokenize and lemmatize a document.

        Args:
            doc (str): Document to be tokenized & lemmatized.

        Returns:
            list: lemmatized word tokens
        """        
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
    """Get a configured TfidfVectorizer.

    Args:
        lemmatize (str, optional): Type of lemmatization. Must be 'default',
        'custom', or 'none'. Defaults to 'default'.

    Raises:
        ValueError: if `lemmatize` is not one of the above.

    Returns:
        TfidfVectorizer: a configured TfidfVectorizer.
    """
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
        """Instantiate a TfidfPredictor. A TfidfPredictor vectorizes input
        documents, then compares them against a training set of documents,
        returning labels corresponding to the most similar documents in the
        training set.

        Args:
            lemmatize (str, optional): See `LemmaTokenizer`.
            n_best (int, optional): See `VectorSimilarity`.
            label_names ([type], optional): Names of each label to be trained
            on, in the same order as the labels appear in the training set.
            Required for returning API responses.
        """
        self._vectorizer = get_vectorizer(lemmatize=lemmatize)
        self._similarity = VectorSimilarity(n_best=n_best)
        self._pipe = make_pipeline(
            self._vectorizer,
            self._similarity
        )
        self.label_names = label_names

    def fit(self, corpus, labels, verbose=False):
        """Fit the TfidfPredictor to a training corpus and labels. This is an
        in-place operation.

        Args:
            corpus (Iterable[str]): Corpus of training documents.
            labels (Iterable[any]): Labels corresponding to each of the training
            documents.
            verbose (bool, optional): Whether to print the training time.
            Defaults to False.
        """        
        start = time()
        self._pipe.fit(corpus, labels)
        print('Training took', time() - start, 'seconds') if verbose else None

    def predict(self, X, verbose=False):
        """Predict the most similar documents in the training set.

        Args:
            X (Iterable[str] | str): Document(s) to be compared against the training set.
            verbose (bool, optional): Whether to print the prediction time.
            Defaults to False.

        Returns:
            (np.ndarray, np.ndarray): the n best labels and dot product scores
            from the training set.
        """        
        start = time()
        if type(X) == str:
            X = [X]
        pred, score = self._pipe.predict(X)
        print('Prediction took', time() - start, 'seconds') if verbose else None
        return pred, score
    
    def predict_obj(self, X, verbose=False):
        """`predict()`, but return a dict (i.e. object) for an API response.

        Raises:
            ValueError: if more than one document is passed, the label names are
            not configured, or if the number of labels does not match the labels
            provided during training.

        Returns:
            dict: Prediction and score object.
        """
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
        
        
        # Package predictions into a dict
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
        """Determine the most unique vocab words in a document.

        Args:
            doc (Iterable[str] | str): A document to be inspected.
            n_top (int, optional): Number of vocab words. Defaults to 10.

        Raises:
            ValueError: if more than one document is passed.

        Returns:
            list[tuple[str, float]]: The most unique vocab words in the document,
            along with their dot product scores.
        """        
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
        """For each word in the query, determine the weight of that word in the
        document.

        Args:
            query (str): Query.
            doc (Iterable[str] | str): Document.

        Raises:
            ValueError: If more than one document is passed.

        Returns:
            list[tuple[str, float]]: The tokenized query, along with the dot
            product score of each word from the document.
        """
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
