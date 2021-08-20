# Week 06 (July 12-16, 2021)

## To Do’s

* Take down emr clusters
* Reach out to SageMaker person who helped with ticket
* Update Project Doc
* Implement sklearn pipeline
    * Implement CosineSimilarity
    * Implement TF-IDF
    * Implement Tokenizer
    * Implement pipeline


## Questions

* Matt: Could I get an assigned desk in Alexandria?
    * Asked Tracy about it
* How do I add env vars for my AWS secret & access keys?

## Friday, July 16, 2021

* Interesting problem: some people like to just post issues with a single screenshot
* Vectors are 85,000 dimensional... yikes
    * Could we do PCA?
* May need to either
    * Switch to just modeling individual repos
    * Reduce dimensionality via PCA or SVD
    * Clean up data on a per-repo basis
    * Switch to object2vec

## Thursday, July 15, 2021

* Standup
    * Any GH Issues that stand out?
        * Redux and Next JS
    * Docs
        * Now people want to know how docs are being improved
        * Now going to be presenting every Friday
    * Something that’s come up: Someone wants to buy ad space on StackOverflow for Amplify, but the ads weren’t being generated correctly
        * Want to improve the SEO of advertisements
* **We should see whether a certain issue is disproportionately more related than other issues**
* **We can add a new function to the Pipeline object that is just predict_score**

## Wednesday, July 14, 2021

* Intern Circles Meeting
* 1:1 with Matt
* Implemented pipeline
* Cleaning up VectorSimilarity
    * Should implement caching for the linear_kernel, since the training vectors will likely grow tremendously in size

## Tuesday, July 13, 2021

* Offboarding for Echo Frames study

## Monday, July 12, 2021

* 1:1 with Mike
    * Updating on switch from PySpark to sklearn
* Midpoint check-in with Mike & Matt
    * Feedback: Take a scaled approach for the future, and make sure that my solutions will work even when I’m gone
* **Frontier has agile seating:** floors 5-7, 10-16
* Implementing VectorSimilarity Estimator
    * check_estimator() is your best friend
    * Implemented fit() and predict()
    * Tested on simple examples

