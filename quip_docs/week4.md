# Week 04 (June 26-July 2, 2021)

## To Do’s

* Get basic GitHub Action up and running in own repo
* Export vectorized data into S3
* Make design diagram: https://design-inspector.a2z.com/
* Update project doc

## Questions

* Can we use some sort of tabular data store rather than object based?
    * Yes, eventually we will be moving away from S3 to some other storage solution
* Should I be committing my SageMaker code to a GH repo?
    * Yes.

## Friday, July 2, 2021

* Working on GitHub Actions
    * Turns out that we can get the issue body and number directly
    * Managed to set up a GitHub Action for grabbing said info
* Want to set up endpoint on SageMaker to make inferences
    * Can take in GET requests with data
    * GH Actions can easily handle this using `curl`

## Thursday, July 1, 2021

* Standup
    * Update: managed to get data into Python, will be working through that today. Started a bit on a design diagram - are there any good examples that I could look at?
* Wrote some more code for making individual/bulk inferences
    * Unable to run code, as PySpark kernel requires an EMR cluster
    * Wrote basic tokenize, vectorize, and comparator functions

## Wednesday, June 30, 2021

* 1:1 with Matt
    * Talked about blocking issues
    * Matt reminded that I should ask around in Slack if possible
* Figured out how to look at S3 logs
* 1:1 with Mike
    * Updated him on my blockers, told him that I was planning on working on setting up the GH Action
    * S3 should be a raw data lake, but I can choose what downstream storage I use (e.g. Redshift)
    * Under the assumption that we’re settling on TF-IDF:
        * Have a weekly cron job that pulls all new issues, vectorizes them, and places them into the issue catalog. Cross-compare all of these new issues
        * When someone files a new issue, we vectorize and compare it against the issue catalog without putting it in the catalog
* Figuring out how to upload vectorized data from S3 to a Jupyter Notebook
    * AlgorithmError arose because CSV doesn’t do well with parquet structs
    * Resolved by switching output file type from CSV to parquet
    * Finally managed to pull vectors into Jupyter!
* Researched how to implement the first step of puzzle: issue creation → inference step
    * Seems like GH Actions is not the best idea, since it doesn’t allow us to know which issue triggered the action
    * This looks like a better flow, AWS API Endpoint + GH Webhook + Lambda: https://aws.amazon.com/quickstart/architecture/git-to-s3-using-webhooks/

## Tuesday, June 29, 2021

* Sick day
* Intern Circles

## Monday, June 28, 2021

* Check-in with Mike
    * Going over what I’ve done so far with trying to get the vectors out of Data Wrangler and into my IPYNB
    * Using S3 and Glue jobs
* Trying to use Glue to concatenate parquets together
    * Seems that Glue doesn’t have a built-in concat function that can do it over multiple files
    * Will possibly need to find another way to combine them
* Coffee chat with Adam
* Troubleshooting the S3 export job
    * Fixed `get_execution_role()` error, due to `sagemaker` Python package being an old version
    * Fixed an error due to parquets having names longer than 64 characters
    * Can’t figure out how to find the job logs for processing job, will work on that later
* Writing some other code for processing vectors that would be useful later on

