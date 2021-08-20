# Intern Project Report: Amplify Issue Librarian

**Discovering similar issues across Amplify’s GitHub repositories**

Intern: Oliver Leung
Mentor: Michael Edelman
Manager: Matt Auerbach

## Context

As of August 2021, users have opened about 18,000 issues in total across AWS Amplify’s 25 repositories, of which 3,000 are still open. Being an open-source project and given the number of repositories, this is not a bad situation to be in, although there is room for improvement. For one, there can be overlapping bug reports or feature requests across multiple libraries - a developer facing a problem with the Android SDK may not know of a potential fix that was found in the JavaScript library. GitHub’s built-in issue search functionality also leaves much to be desired, as it’s limited to only one repository at a time and gives subpar search results. It then becomes a Herculean effort for a developer to keep track of all of the issues coming in each week and to close existing ones that are similar/duplicates.

## Goal

When an Amplify customer opens a new GitHub issue, they should be notified if there exist any sufficiently similar, previously opened issues across the `aws-amplify` organization, so that they can quickly access such issues and resolve the problem that they are facing. Furthermore, when an Amplify developer would like to submit search queries across all of Amplify’s GitHub issues, they should be able to do so through the use of an internal tool.

## Example Usages

### Opening a new issue

Here is an example of a customer opening a GitHub issue and receiving a GitHub Actions bot response, with a list of potentially similar issues.

### Searching for issues

Here is an example of a developer entering in a search query in the [internal search tool](https://oliver-leung.github.io/amplify-observer-nlp/).

## Example Queries

* datastore subscription model fails
* password manager autofill
* image file upload fails
* DataStore @auth sync error subscription failed Missing field argument owner

# NLP Methodology

At a high level, I wanted to create *abstract representations* of GitHub issues, and then compare those representations to find similarities. As with many NLP and information retrieval solutions, these representations will take on [the form of vectors](https://medium.com/@paritosh_30025/natural-language-processing-text-data-vectorization-af2520529cf7) that can be quantitatively compared. Since the data is unlabeled (and manually labelling similar issues would defeat the purpose of the project), we’ll also be working under an *unsupervised learning* paradigm.

In this methodology section, I explain my finalized approach to processing the training data, training the model, and running inferences. Then, I summarize the NLP-related insights that I gained through working on this problem, which ultimately guided the approach that I settled on.

## Training Data Pre-Processing

 Here is a contrived example and explanation to illustrate my data pre-processing pipeline.

|Explanation	|Example	|
|---	|---	|
|The raw, unedited training data comprises of GitHub issues from Amplify repositories, for each of which we extract the title, body, and URL.	|_URL:_ github.com/aws-amplify/amplify-nlp/issues/42069

_Title:_ This is a new issue, with a very cool title

_Body:_ 

# Describe the issue that you are facing.

This is the body of the issue. Often times, customers will use Markdown to format their issues, with *italics*, _underlines_, or ~strikethroughs~. This example is in pre-rendered Markdown.

# Describe any potential solutions that you've tried.

They may also provide URL's, either [embedded](aws.amazon.com/amplify/) or in plaintext: aws.amazon.com/amplify/ . 

```
console.log("It's common to see code blocks or error messages for implementation questions.")
```	|
|Code blocks, URL's, and Markdown formatting characters are removed from the body.	|_URL:_ github.com/aws-amplify/amplify-nlp/issues/42069

_Title:_ This is a new issue, with a very cool title

_Body:_ 

# Describe the issue that you are facing.

This is the body of the issue. Often times, customers will use Markdown to format their issues, with italics, underlines, or strikethroughs. This example is in pre-rendered Markdown.

# Describe any potential solutions that you've tried.

They may also provide URL's, either embedded or in plaintext: 	|
|Using an intermediate AST representation, issue template headers and boilerplate text are filtered out.	|_URL:_ github.com/aws-amplify/amplify-nlp/issues/42069

_Title:_ This is a new issue, with a very cool title

_Body:_ 

This is the body of the issue. Often times, customers will use Markdown to format their issues, with italics, underlines, or strikethroughs. This example is in pre-rendered Markdown.

They may also provide URL's, either embedded or in plaintext: 	|
|The title and body are concatenated to form the title-body, which is the example instance (i.e. input data to the model). The title and URL are used as the example label (i.e. output data from the model).	|_Title-Body:_ This is a new issue, with a very cool title

This is the body of the issue. Often times, customers will use Markdown to format their issues, with italics, underlines, or strikethroughs. This example is in pre-rendered Markdown.

They may also provide URL's, either embedded or in plaintext:

_Title, URL:_ 
     This is a new issue, with a very cool title
     github.com/aws-amplify/amplify-nlp/issues/42069	|

## Model Training & Inference

### Tokenization, POS Filtering, Lemmatization

After pre-processing the text, I had a few more steps to take before feeding the training data to the model:

1. *Tokenization* is the process of breaking up text into individual word tokens that can then be processed by a model. My tokenization strategy was quite simple: find all sequences of at least 3 letters. Tokens were also made to be all lower case.
2. I used [NLTK’s part-of-speech tagger](https://www.nltk.org/book/ch05.html) to map each word token to a POS tag. I then used those tags to *filter out non-verb and non-noun words*. This is similar to *stop word* *removal*, a common NLP strategy which removes words such as articles or prepositions that don’t carry unique semantic “sense” (e.g. “a”, “the”, “with”).
3. *Lemmatization* is the process of [removing case and tense markers](https://en.wikipedia.org/wiki/Lemmatisation) from word tokens. This is useful because it simplifies the vocabulary by grouping together semantically identical words of different inflections (e.g. “walk”, “walks”, and “walking” all become *“walk”*). I used the POS tags from the previous step to detect the part of speech of inflectional forms, which aided [WordNet’s lemmatizer](https://www.nltk.org/_modules/nltk/stem/wordnet.html) in finding the base word form.

TODO: Diagram and explanation are out of sync

```
# Tokenization pipeline visualization

Bees are making honey in the garden
||
|| Basic Tokenization
\/
bees, are, making, honey, the, garden
||
|| Lemmatization
\/
bees, be, make, honey, the, garden
||
|| Stop Word Removal
\/
bees, make, honey, garden
```

### TF-IDF Document Vectorization

[Term Frequency-Inverse Document Frequency (TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a simple, yet popular algorithm in information retrieval and search engines. It’s based on the assumption that, if a word appears frequently in one document, but doesn’t appear in many other documents, then that word must be highly representative of that document. For each token in each document, TF-IDF is computed as the product of the two following metrics:

For a token *t* in a document *d*, the *term frequency* is the number of times *t* appears in *d*, divided by the total number of tokens in *d*.
[Image: image.png]For a token *t* and a corpus of documents *D*, the *inverse document frequency* is the logarithm of the number of documents in *D* divided by the number of documents in which *t* appears at all.
[Image: image]I used Scikit-Learn’s `TfidfVectorizer` to vectorized the tokenized documents. When the vectorizer is initially fitted to the corpus, it creates an ordered list of each unique word in the corpus (i.e. the vocabulary). Then, a matrix of TF-IDF values are computed; the matrix has a dimensionality of (number of documents) x (size of vocabulary). Each row of the matrix corresponds to the vectorized form of a document.

### Cosine Similarity

After mapping a document to this TF-IDF vector space, it’s more intuitive to visualize the vector space by thinking of each word in the vocabulary corresponding to one axis (or basis vector) in the vector space, and then “pointing” a vector towards the axes that have the highest TF-IDF values. For example, if “datastore” gets a high TF-IDF value for a document (i.e. it’s highly representative of the doc), then that document’s vector would point towards the “datastore” axis.

A good quantitative metric of whether a document is similar to a query is the *[cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity)*. This metric is commonly used in information retrieval, especially since it’s computationally fast for sparse matrices - only entries for which both vectors have non-zero values need to be computed. Put simply, the element-wise dot product of two vectors is proportional to the cosine of the angle. Since our `TfidfVectorizer` returns normalized vectors, the dot product is exactly the cosine.

When performing inferences, the search query is tokenized, then vectorized according to the vocabulary and stored IDF values in the `TfidfVectorizer`. The pairwise dot product is then computed against each of the existing document vectors (again, each corresponding to one GH issue), and the issues with the highest cosine are returned as part of the inference.
[Image: image.png]Credits to Towards Data Science: https://towardsdatascience.com/calculate-similarity-the-most-relevant-metrics-in-a-nutshell-9a43564f533e
[Image: Credits to Towards Data Science: https://towardsdatascience.com/calculate-similarity-the-most-relevant-metrics-in-a-nutshell-9a43564f533e]
## Key Insights on the Corpus

Throughout my time working with the corpus of existing Amplify GitHub issues, I observed several key insights that defined my approach to the data**:**

1. **Unknown Words:** Many issues mention classes, functions, or fields that don’t exist in natural language (e.g. `AmplifySignIn`).
2. **Summarization in Titles:** Customers are good at intuitively summarizing the issue in the title, but less so in the issue’s body.
3. **Non-Natural Language:** Some issues include code blocks and URL’s, which are not representative of natural language.
4. **Issue Templates**: Most customer-facing Amplify repositories use issue templates, which are “noisy” and don’t add any information to the data.

The first key insight limited my model choice. The most common algorithm for text vectorization is the use of *word embeddings*: mapping each word token to pre-determined, low-dimensional vectors, and then averaging together every token’s vector within the text. However, word embeddings rely on a fixed vocabulary, which make them hard to use with text consisting of many novel tokens (such as code). I decided to instead go with *Term Frequency-Inverse Document Frequency (TF-IDF)*, which is quite flexible when given data with an unusual vocabulary.

The second key insight determined how I chose my training and inference outputs. My initial approach was to train the model on issue bodies, and then perform inferences by comparing a new issue’s body against all of the existing issues. However, I found that the model didn’t mark certain similar pairs that I was expecting, because the customer would *summarize many keywords of the issue within the title*, but not within the body. Furthermore, comparing issue bodies with other issue bodies introduced lots of noise for my inference inputs. To overcome these difficulties, I trained the model on the *concatenation of the issue title and body*, then performed *inferences with smaller search queries or the titles of new issues*.

The third and fourth key insights informed my pre-processing strategy. During early testing, I noticed that *code blocks and error messages* would heavily manipulate my inferences, due to certain keywords being highly repetitive (skewing the TF metric). Furthermore, some users would provide *URL*s in their issues, which are also not very representative of natural language. And although TF-IDF would ignore the *issue template headers* in theory (since they were in almost every issue and would be accounted by the IDF metric), they introduced unnecessary information to the data. Parsing out these three elements led to moderate improvements to both accuracy and training speed. (Big shoutout to Mike for helping me a ton with these steps!)

## Key Insights on Model Performance

As I was optimizing the model, I also observed several key insights that helped improve the model’s accuracy and speed.

1. **Well-defined Features:** Upon vectorizing a document, each feature (i.e. TF-IDF value) in the vector corresponds to one word in the vocabulary, making it easy to determine which words are unique to a given document.
2. **Managing the Feature Space:** Since the size of the vocabulary corresponds directly to the size of the feature vectors, it was very important to cut down on useless words in the vocabulary.

Because of the first key insight, I could easily troubleshoot the model’s results. I wrote inspection functions that allow one to check the words with the highest TF-IDF value within a given document, as well as query a document for its TF-IDF values of selected words. I was able to better understand and visualize how documents were tokenized and vectorized, which informed how I pre-processed the data.

Acting on the second key insight was perhaps the most important task in improving the model’s performance. The early, naive implementation had a vocabulary of over 65,000 words, took at least 5 minutes to train, and took about 10 seconds to perform inferences. Moreover, the model needed 65,000 TF-IDF values for each of the 17,000 GitHub issues; NumPy arrays with over 1 billion floats are far from ideal, and led to memory allocation issues. But after gradually introducing refinements in the form of preprocessing, lemmatization, and POS tagging, I managed to cut the vocabulary down to 21,000 words, making the training time about 80 seconds and inference time about 2.5 seconds. Being able to quickly train and test the model facilitated the process of validating the model and ensuring that it gave reasonable prediction results.

# Project Architecture

[Image: Amplify Observer.png]
## Inference Pipeline

Inferences are performed in a fairly straightforward stack of components; this section goes over each of those components from the bottom up.

### Model Implementation and Deployment

Each part of my TF-IDF model uses SKLearn’s [Estimator](https://scikit-learn.org/stable/developers/develop.html) framework for its various features (like compatibility with SKLearn Pipelines) and for consistency. First, I implemented a *LemmaTokenizer* class to perform lemmatization and tokenization of input documents. Next, I utilized SKLearn’s built-in *[TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)* class to vectorized the tokenized documents. More specifically, *LemmaTokenizer* is passed as a parameter to *TfidfVectorizer*, the latter of which actually runs the tokenization step. Finally, I implemented a *CosineSimilarity* Estimator for computing the pairwise cosine distances for each of the vectorized documents.

SKLearn’s [Pipeline interface](https://scikit-learn.org/stable/modules/compose.html) is a great way of chaining together multiple steps within a ML or Data Science flow, and provides good abstractions for both fitting a model and running inferences on it.

The trained model is then deployed to a [SageMaker Inferences Endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html), a common feature of SageMaker that allows for general clients to perform inferences on a model. However, the model is still only callable from within a Jupyter Notebook in SageMaker.
[Image: image.png]
### Backend Request Handler Implementation

In order for the model to be used in more contexts, I implemented a simple REST API endpoint in API Gateway. The endpoint triggers a Lambda Function, which handles the request and forwards it to the SageMaker Inferences Endpoint. Since the output schema of the model is more generified, I handled the specific output formatting for each application (GitHub bot and search engine) within two separate endpoints.

### Frontend Interface Implementation and Deployment

I configured a GitHub Actions workflow that would trigger every time a customer opened a new issue. It would send the title of the issue as a search query to the API endpoint, and then reply to the customer with the top most similar issues across the Amplify Org.
[Image: image.png][Image: image.png]The beauty of this implementation is that, if one wants to deploy the bot to a new Amplify repository, the process is as simple as copying the workflow configuration YAML (above left) to the new repository.

## Training Pipeline

The raw data in the form of Amplify GitHub issues is retrieved using GitHub’s API, and then data pre-processing is run in a Glue job to extract just the important information from the data. Then, the pre-processed data is stored in an S3 bucket.

SageMaker’s [Script Mode](https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode/) is a powerful tool for using a custom model algorithm (as I did above) and training it on a SageMaker instance. After implementing my model and preparing the training data, I run the following script within a SageMaker Notebook Instance to download the S3 data, train the model, then compress and store the model in a SageMaker-specific S3 bucket.

```
# An excerpt from my training script that downloads training data and fits the model to the data.
# Note that some referenced functions are not shown.
args = parse_args()
    
# Ensure that these match the hyperparams specified in parse_args.
hyperparams = {
    'n_best': args.n_best,
    'lemmatize': args.lemmatize,
    'label_names': ['Url', 'Title']
}

# Select corpus and labels. Note that the column labels specified below and above are specific to the dataset being used.
train_df = load_files(args.train)
train_X = train_df['title_body']
train_y = list(zip(train_df['url'], train_df['title']))

# Create and train pipeline
clf = TfidfPredictor(**hyperparams)
clf.fit(train_X, train_y)
print(clf.label_names)

# Print the coefficients of the trained classifier, and save the coefficients
joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
```

Script Mode also allowed me to configure the exact serialization/deserialization process for the input and output of the model, which was helpful for handling requests from different sources. After training the model, I have the option of deploying the model to an inferences endpoint.

There is still room for improvement with the training pipeline. We still need to manually download the data, run the Glue job, and run the training script each time that we want to retrain the model, which is critical for ensuring that it stays up to date with the latest issues. With my remaining time this summer, I’m planning on starting up a Step Function that will automatically run the data processing and training pipelines each week.

# Future Steps

## Improving Computation Time with Numba

Although cosine similarity is fast to compute with sparse vectors (i.e. vectors with many zeros, like TF-IDF vectors), there are many GitHub issues to compare, and the number keeps growing, Using Numba, I could easily parallelize the cosine similarity computations, and possibly the training process itself.

## Improving Accuracy with Neural Networks

In 2018, Google unveiled BERT, a state-of-the-art neural network model for encoding/decoding natural language. BERT is very powerful in unsupervised settings, but is difficult to refine since features are not well-defined, and NN training is computationally intensive. Another possibility is using object2vec, which uses novel methods to turn unsupervised information retrieval methods into a supervised learning model.

## Recording User Feedback

Currently, there is no way of telling whether a customer likes or dislikes the predictions provided by the model. One potential way to get over this is to store each inference that is made, and then to instruct the customer to thumbs-up or thumbs-down react to the GitHub bot reply depending on how helpful the prediction was. This would necessitate some extra services to be spun up in order to store the customer responses, which could then be analyzed internally.

## Fix Lambda Spin-up

This was a problem that showed up in my final presentation, where the GitHub bot would call the API Gateway which would call the Lambda, but the Lambda was dormant and returned an error message instead of calling the SageMaker inferences endpoint.

# Specific Service Instances Used

This is a (hopefully) comprehensive list of all the services and instances I spun up for the project, to ensure easier cleaning and transfer to other accounts.

## SageMaker

* Studio Notebook - `oliver`
    * Where I did most of my work. Contains all of my code, and is connected to my GitHub repository. You can run basically every part of my project in there.
* Notebook instance - `oliver-issue-similarity`
    * Though I didn’t use this that much, it’s useful if you want to run the training script more quickly, as it uses an already-running SM instance
* Training Jobs, Models, Endpoint configurations
    * These are really just logs of each time I trained/deployed the model.
* Endpoints
    * `issue-similarity-endpoint` serves the trained model after it’s trained and deployed through the Studio Notebook or Notebook Instance. It’s connected to the Lambdas that serve the API endpoints (on the Lambda side).
    * `test-endpoint` was made when we were still messing around with our deployment and API config - it can be safely deleted.

## Lambda

* `amplify-github-issue-similarity` is an API request handler that calls the `issue-similarity-endpoint` SageMaker endpoint, then reformats the model’s issue prediction output so that the GitHub bot prints a nice response.
* `invoke-issue-similarity-endpoint` is a generified API request handler that returns the predicted issues in a JSON format.

## API Gateway

* `AmplifyGitHubIssueSimilarity` provides a POST route that calls the `amplify-github-issue-similarity` Lambda and serves the GitHub bot.
* `IssueSimilarityPrediction` provides a POST route that calls the generified `invoke-issue-simiarity-endpoint` Lambda.

## S3

* `amplifyobserverinsights-aoinsightslandingbucket29-5vcr471d4nm5` contains the pre-processed training data in the `data/issues/` directory. These are loaded when model training occurs.
* `sagemaker-us-west-2-092109498566` contains the artifacts and (hyper)parameters of the trained models. The SageMaker inference endpoint deserializes models from this bucket in order to run inferences.

# How to change stuff down the line

I’ve stored my code in this GitHub repo: https://github.com/oliver-leung/amplify-observer-nlp. It may be private at the moment, but you can ask Michael Edelman/Matt Auerbach to give you access if that’s the case.

## Configuration changes

### I want to change the hyperparameters (e.g. return more than 10 similar issues at a time)

In the 3rd code cell of `sklearn-tfidf-script-mode.ipynb`, edit the `hyperparams` dictionary, then run all code cells in SageMaker Studio.

### I want to use training data from a different bucket/folder

In the same code cell, edit `train_data` with the S3 URI. If the data is in a subfolder, make sure that you add the subfolder path to the end of the URI.

### I want to deploy a model with different hyperparameters to a different endpoint

If need be, create a new endpoint through the SageMaker GUI. Then, in the 4th code cell of `sklearn-tfidf-script-mode.ipynb`, change the `EndpointName` parameter of the final call to `client.update_endpoint` to whatever you named the other endpoint.

### I want the model to return a different set of information about the issues

In the `train_tfidf.py` training script, change `train_y` to use a different set of columns from the training data. The existing values are hard-coded to match the column names in the current training data.

### I want to deploy the model to a new Amplify repo

Copy `.github/workflows/main.yml` to the same path in the Amplify repo (doesn’t have to be called `main` specifically).

## Implementation changes

Most of these changes are in `tfidf_predictor.py`.

### I want to change the lemmatization/tokenization step

Edit `LemmaTokenizer.__call__`.

### I want to change the JSON output schema

Edit `TfidfPredictor.predict_obj`. Note that it depends on the `TfidfPredictor.predict` function.

### I want to change the model algorithm from TF-IDF to something else (e.g. BERT)

First of all, how dare you! Kidding. After you’ve chosen what class the model will use, edit the training data preparation and hyperparameters inside of the `if name == "main"` section in `train_tfidf.py`. Then, in the 3rd cell of `sklearn-tfidf-script-mode.ipynb`, edit the parameters, and switch the `SKLearn` object to whichever 3rd party library you’re using. More specific instructions can be found [here](https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode/).

## Random notes

* The training script `train_tfidf.py` and the training script *runner* `sklearn-tfidf-script-mode.ipynb` are tightly coupled - any changes to configurations, parameters, and implementations should be checked across both files.
* `sklearn-tfidf-poc.ipynb` is a good place to learn how the model algorithm works, but is not a good representation of how it’s trained and deployed in script mode - I often had to use quite different tools/classes/functions to achieve similar things.
* `utils_tfidf.py` is a bit messy and non-generic, but it should only be used with the `sklearn-tfidf-poc.ipynb` notebook for the most part.
* `VectorSimilarity` in `tfidf_predictor.py` computes cosine similarity and uses a few pretty dense NumPy operations. You shouldn’t ever need to change it, but if you do, do so at your own risk!

`[amplifyobserverinsights-aoinsightslandingbucket29-5vcr471d4nm5](https://s3.console.aws.amazon.com/s3/buckets/amplifyobserverinsights-aoinsightslandingbucket29-5vcr471d4nm5?region=us-west-2)
