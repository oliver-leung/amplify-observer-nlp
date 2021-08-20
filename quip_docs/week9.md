# Week 09 (Aug 2-6, 2021)

## To Do’s

* Critical Path:
    * Figure out SageMaker local mode
    * Fix model output schema
    * Clean up GitHub bot output
    * Set up internal search tool for debugging
    * Post public testing doc
    * Write documentation for functions
    * Clean up Script Mode NB
* Non-technical:
    * Update Project Doc
    * Prepare presentation
    * Complete Intern Self-Review - **Due Mon, Aug 16th, 2021**
* Nice to haves:
    * See whether new endpoint updater works **No, it’s way too slow**
    * Merge API routes into single endpoint
    * Record thumbs up or thumbs down
    * Improve computational time with NumPy record arrays and joblib parallelization
        * https://numpy.org/doc/stable/user/basics.rec.html
    * Set up automated retraining workflow in Step Functions

## Questions

## Thursday/Friday, August 5-6, 2021

* Set up internal search tool
    * Difficulties with API CORS, but eventually fixed them up
    * Also set up feedback doc
* Writing documentation for my code
* Clean up script mode NB

## Wednesday, August 4, 2021

* Chat with Matt
    * Updating on project progress
    * Scheduling final presentation
* Fixed GitHub issue output

## Tuesday, August 3, 2021

* Chat with Mike
    * Don’t worry about improving speed with JIT
    * Should start preparing for presentation, meeting anyone else new at Amazon
* Working on getting JSON output
    * Difficulty with deploying to endpoint

## Monday, August 2, 2021

* Trying to follow these steps to mount EFS on SM Notebook instance, but no luck:
    * https://stackoverflow.com/questions/38632222/aws-efs-connection-timeout-at-mount/57141195
    * Figured it out by copying the Security Group rules for inbound/outbound that were automatically created for connecting SM Studio Notebooks to EFS isntances:
    * Useful guide: https://aws.amazon.com/blogs/machine-learning/mount-an-efs-file-system-to-an-amazon-sagemaker-notebook-with-lifecycle-configurations/
* Managed to deploy the model with huge improvements to speed and without load errors
* Trying to implement Numba JIT compiler, but not much luck

