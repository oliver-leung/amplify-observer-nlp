# Week 03 (June 21-25, 2021)

## To Do’s

* Complete SDE Bootcamp
* Look into Cloud Practitioner
* Complete AWS Full Stack tutorial: https://aws.amazon.com/getting-started/hands-on/build-web-app-s3-lambda-api-gateway-dynamodb/?e=gs2020&p=fullstack&p=gsrc&c=lp_fsd

## Questions

* Conditional Random Fields - what the hell are they?
* Ask Matt about evaluation process

## Thursday, Friday

* Finishing up BT101... Had a headache and couldn’t work as much
* Learning how to use SageMaker Data Wrangler
    * Figured out how to do concatenation and TF-IDF vectorization directly in Data Wrangler!
    * Still need to see if we can cut out extraneous template sentances (e.g. “Describe what the problem is”), but in theory, TF-IDF will not care about them because they’re so common across the data
* Learning how to export Data Wrangler data to a Jupyter Notebook
    * Option 1: Python Code - supposedly least overhead, but requires an EMR cluster. Maybe I could get that?
    * Option 2: S3 bucket

## Wednesday, June 23, 2021

* Tried to load s3 JSON’s into a SageMaker IPYNB, but kept getting an authorization error
* Looking for other resources on SageMaker
* Sprint Planning
    * Reflecting on how on-call is going for TJ
        * On-call is usually pretty lowkey
    * Sprint Planning Meeting
        * Trying to figure out how to combine docs
    * Metrics Dashboard has been having some trouble over the last two wks
    * Discussing the Issue analyzer task
        * Can use linked issues to determine whether they are relevant
* Learning Sagemaker
* It seems that TF-IDF is the best approach (open in Incognito): https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05
* Data Wrangler has a built-in TF-IDF vectorizer: https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html#data-wrangler-transform-featurize-text
    * Might wanna use sklearn though, because we’re tokenizing code names
* SageMaker also has Object2Vec for comparing document embeddings: https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_applying_machine_learning/object2vec_document_embedding/object2vec_document_embedding.html
    * https://aws.amazon.com/blogs/machine-learning/introduction-to-amazon-sagemaker-object2vec/
    * Pros: we can automatically turn this into a supervised learning task
    * Cons: users generally don’t write that much in the issues

## Tuesday, June 22, 2021

* Try using Cloud Guru

## Monday, June 21, 2021

* Continuing setup of BuilderTools and Cloud Desktop
* Reading NLP2Code paper: https://arxiv.org/pdf/1701.05648.pdf
    * Main idea: Users enter in a description of a code snippet, then NLP2Code finds the relevant code snippet from Stack Overflow and pastes it directly in the IDE
    * Users can also cycle through sets of potentially relevant snippets
    * TaskNav algorithm is used in order to map from generalized coding concepts to specific coding tasks (“content assist”)
        * Paper: https://www.cs.mcgill.ca/~martin/papers/tse2015b.pdf
        * https://www.cs.mcgill.ca/~swevo/tasknav/
        * 1) User begins typing in a programming concept (e.g. “time zone”). 2) TaskNav presents potential tasks relating to the concept (e.g. “convert between time zone”). 3) TaskNav points users to instructions on implementing said task.
    * After TaskNav presents Stack Overflow answers, NLP2Code extracts the top few code snippets from the relevant thread
    * **Overall: This would be a useful method for Amplify users to find code snippets from documentation, but TaskNav is now defunct and was only useful for the Java API when it was still around. If we wanted to apply it to our work, we would have to re-implement it and apply it to the Amplify Docs.**
* Reading “Classifying SO Posts on API Issues“: https://www.cs.usask.ca/~croy/papers/2018/SANER2018/AhasanuzzamanSANER2018SO_API.pdf
    * Main idea: Use a supervised learning model to classify whether SO posts are relating to API issues
        * Relating to the problem of figuring out whether GH issues are because the user doesn’t know how to implement something, or because of a legitimate problem with the API/Docs
    * Higher-reputation users on SO tend to ask more API issue-related questions
    * LDA model was used to find topic word distribution
    * Step 1: Sentence extraction, text transformation
        * Removing HTML tag garbage, extracting sentences in the absence of sentence end characters
        * PROBLEM_CODE vs. NORMAL_CODE - used a regex to first filter out stack traces, then marked other “incorrect code” PROBLEM_CODE
    * Step 2: Issue sentence identification
        * Assumption: Issue-related posts contain issue-related sentences
        * CRF requires **manually-annotated issue sentences** for training; may not be scalable for my internship
        * Kind of got lost after this....
    * **Overall: This would be a useful method to differentiate between help issues and API bug issues, but like Matt mentioned, work with auto-labeling GH issues is inherently low impact, given that our corpus is a few orders of magnitude smaller than the corpora used in the paper.**

