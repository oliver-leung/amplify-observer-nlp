# Amplify Librarian - Brainstorming

## Problem

Amplify repos tend to have many feature request issues, [ranging in the hundreds](https://github.com/aws-amplify/amplify-js/issues?q=is%3Aopen+is%3Aissue+label%3Afeature-request) a lot of the time. However, not all of these issues are useful - some of these feature requests are duplicates, and others have already been implemented and deployed, but the respective issue was not closed. It would take an unreasonably long time for a developer to go through and clean out the duplicates one-by-one, and this process would have to be repeated many times in the future.

# End Goal

We want to analyze feature request issues and pull requests to determine which ones are duplicates of each other, so that a developer can know which ones to close without having to read all of them.

# Ideas

* Compare issues against documentation pages
* [Document embedding techniques](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d)

# How to get there

## Document retrieval


### Methods

For now, we will refer to one *document* as one issue/PR and all of its comments, and the *corpus* as the set of all documents. One alternative is to define each comment within an issue/PR as a document, but given the number of comments an issue can have, this would greatly slow down our pairwise comparisons.

We will be using the GH API to [list out issues](https://docs.github.com/en/rest/reference/issues#list-repository-issues) with the `feature-request` label, as well as PR’s. Then, we will store the issues’ content and number in tabular form (which database service?).

### Challenges

The total number of issues/PR’s (either open or closed) is strictly increasing. Clearly, we have to examine both open and closed issues to ensure that we don’t have an older duplicate. But, how can we manage the scale of both the amount of documents in our corpus, and the computations required to compare all of the documents?

While we could cache similarity metrics between each document, these metrics could change if comments are added to an issue/PR. Furthermore, adding the nth document would require n-1 comparisons to all of the old documents.

All of this hinges on the assumption that we’re doing pairwise comparisons across all documents. Is there a better system that could reduce the number of documents to compare?

## Document pre-processing


### Methods

The comments for each issue/PR should be concatenated together. We then need to “homogenize” the data by removing special characters, removing code blocks, tokenizing, then removing stop words (e.g. articles, “this”, “which”...).

Most of these steps can be done with common Python NL libraries like `nltk` or `sklearn`.

This could run on a Lambda?

### Challenges

Following from above, should we be reprocessing a document every time a new comment is added?

Users frequently reference classes and functions, but where most users talk about `X.y()`, a few others abbreviate to just `y()`. This is an edge case, but is there a way to process document word tokens that recognizes when abbreviations are used?

Class or function names are obviously not used in natural language, which locks us out of using semantics-based methods like word embeddings.

## Document vectorization & comparison


### Methods

As with many NLP problems, we want to eventually arrive at a vectorized representation for each of the documents that we want to analyze. The obvious choice would be to use term frequency, inverse document frequency (TF-IDF) to determine which words are most relevant to each document, and return a sparse vector. Then, we can use cosine similarity or other similarity metrics to compare each document with each other.

Again, this can be done with common Python NL libraries like `nltk` or `sklearn`. `text2vec` provides high-level functionality for comparisons.

This could run on a Lambda?

### Challenges

`O(n^2)` runtime isn’t great. How can we optimize this further?
