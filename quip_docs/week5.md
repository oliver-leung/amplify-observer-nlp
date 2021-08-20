# Week 05 (July 6-9, 2021)

## To Do’s

* Make design diagram: https://design-inspector.a2z.com/
* Update project doc

## Questions

## Friday, July 9, 2021

* Intern Circles meeting
    * Planning out agenda for Thu
* Trying to run the PySpark kernel on EMR
    * No luck, commands still don’t work (aside from the “Connect to cluster” command)
* **If we have to manually load files into Data Wrangler, then it doesn’t make sense in the long-term to use it**
    * Thus, it doesn’t make any sense to use PySpark for our vectorizer, nor EMR
    * Thus, I should move over to sklearn
* SKlearn has Pipelines - I could do a tokenize → vectorize → cosine-sim pipeline
    * Would have to briefly implement the last estimator
* Current plan:

    1. Use SageMaker’s [SKLearn estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html), which requires a training script
    2. Write a training script, which uses a pipeline (tokenize, vectorize, cosine-sim)
    3. Implement a cosine-sim predictor

## Thursday, July 8, 2021

* Current assumptions
    * GitHub Actions workflow will be the “user-facing” interface that takes in issue data and replies if dupes are found
    * The GitHub action will send issue data to an inferences API endpoint
    * We want to turn issue text into vectors
    * We want to compare vectors using cosine similarity/dot prod/K-Means
* My options:
    * Implement a new SageMaker class that can perform cosine similarity and be deployed to an endpoint
    * Use existing SageMaker PCA and K-Means models to find clusters of similar documents
        * **Does not work**: like LDA, K-Means requires the number of clusters/topics to be chosen beforehand
    * Move to Lambda and do the same, but without having to implement a new class
    * Abandon TF-IDF and switch over to neural networks completely
* Matt got the EMR cluster up and running
* Chat with Stephen
    * Looks like Lambda is the best option at the moment
    * Don’t worry too much about deployment - if you can talk about your methodology and generalize your work so that it can function with a wider variety of services, that’s better
    * Doesn’t have to be groundbreaking, just needs to tie together existing ideas in a novel way
    * Be sure to document your process
* Going to move towards neural network approach

## Wednesday, July 7, 2021

* 1:1 with Matt
    * Midpoint assessment will be coming soon
    * Updating on what I’ve been working on
* Sprint Planning
    * Updating on docs site
    * I could just search for duplicates manually to test: 
        * https://github.com/aws-amplify/amplify-flutter/issues/626
        * https://github.com/aws-amplify/amplify-flutter/issues/292
* Use K-Means to find clusters of vectors: https://sagemaker.readthedocs.io/en/stable/algorithms/kmeans.html

## Tuesday, July 6, 2021

* Learning more about how to use GH actions
    * Managed to create a simple receive-and-reply workflow for new comments
* Finished making rough draft of my architecture diagram
    * Need some help planning out the model update phase
* Researching Estimator and Predictor classes

