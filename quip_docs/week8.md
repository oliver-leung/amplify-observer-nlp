# Week 08 (July 26-30, 2021)

## To Do’s

* Update Project Doc
* Critical Path:
    * Clean up data storage format in S3
    * Set up SageMaker model deployment to endpoint
    * Connect GH Action workflow to endpoint
    * Set up automated retraining workflow in Step Functions
* Nice to haves:
    * Implement comparator function that finds common words between query and document
    * Improve tokenization
    * Improve computational time with NumPy record arrays and joblib parallelization
        * https://numpy.org/doc/stable/user/basics.rec.html
    * Set up internal search tool for debugging
    * Implement wrapper class for current pipe model
    * Fix list data objs
    * Set up a local training flow in a NB instance

## Questions

## Friday, July 30, 2021

* Dealing with MemoryError
* Need to reorganize filestructure:
    * 

## Thursday, July 29, 2021

* Testing deployment
* Need to return data in format:

`{text: <input>`


## Wednesday, July 28, 2021

* Turns out that just having a large corpus causes inferences to be very slow
    * Probably not able to apply ColBERT to this
* Learning how to use Step Functions and scheduling via CloudWatch/EventBridge
* Got new parquets up and running with huge improvements to speed!

## Tuesday, July 27, 2021

* Resolving errors with training script
    * Seems like the data is not in an ideal format
* What is the ideal format of preprocessed data?
    * Download title, body, and URL from GitHub issues
    * Clean out template info from body
    * Concat title and body
    * Store parquet or CSV with two columns: title-body and URL
* What sorts of inspections would be useful?
    * Inspect single corpus document to see which words are most important
    * Compare query against document to see how important each query word is in the document

## Monday, July 26, 2021

* Should start thinking of how we want to deploy this
    * Option 1: Original idea of feeding potentially similar issues via the GH Actions bot
    * Option 2: Create a basic search engine
* Attempted to add GPU compute, but no luck
* Attempting to set up training flow, but running into errors that take minutes to debug each iteration
* Should look into parallelizing training/lemmatization using joblib
    * https://scikit-learn.org/stable/computing/parallelism.html
    * https://joblib.readthedocs.io/en/latest/parallel.html

## SageMaker training error

```
2021-07-27 00:45:18,905 sagemaker-containers INFO     Imported framework sagemaker_sklearn_container.training2021-07-27 00:45:18,907 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)2021-07-27 00:45:18,916 sagemaker_sklearn_container.training INFO     Invoking user training script.2021-07-27 00:45:19,134 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)2021-07-27 00:45:20,583 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)2021-07-27 00:45:20,595 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)2021-07-27 00:45:20,604 sagemaker-training-toolkit INFO     Invoking user scriptTraining Env:{
    "additional_framework_parameters": {},
    "channel_input_dirs": {
        "train": "/opt/ml/input/data/train"
    },
    "current_host": "algo-1",
    "framework_module": "sagemaker_sklearn_container.training:main",
    "hosts": [
        "algo-1"
    ],
    "hyperparameters": {
        "n-best": 10
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {
        "train": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        }
    },
    "input_dir": "/opt/ml/input",
    "is_master": true,
    "job_name": "sagemaker-scikit-learn-2021-07-27-00-42-38-097",
    "log_level": 20,
    "master_hostname": "algo-1",
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-us-west-2-876895323997/sagemaker-scikit-learn-2021-07-27-00-42-38-097/source/sourcedir.tar.gz",
    "module_name": "train_tfidf",
    "network_interface_name": "eth0",
    "num_cpus": 16,
    "num_gpus": 0,
    "output_data_dir": "/opt/ml/output/data",
    "output_dir": "/opt/ml/output",
    "output_intermediate_dir": "/opt/ml/output/intermediate",
    "resource_config": {
        "current_host": "algo-1",
        "hosts": [
            "algo-1"
        ],
        "network_interface_name": "eth0"
    },
    "user_entry_point": "train_tfidf.py"}Environment variables:SM_HOSTS=["algo-1"]SM_NETWORK_INTERFACE_NAME=eth0SM_HPS={"n-best":10}SM_USER_ENTRY_POINT=train_tfidf.pySM_FRAMEWORK_PARAMS={}SM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"}SM_INPUT_DATA_CONFIG={"train":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}SM_OUTPUT_DATA_DIR=/opt/ml/output/dataSM_CHANNELS=["train"]SM_CURRENT_HOST=algo-1SM_MODULE_NAME=train_tfidfSM_LOG_LEVEL=20SM_FRAMEWORK_MODULE=sagemaker_sklearn_container.training:mainSM_INPUT_DIR=/opt/ml/inputSM_INPUT_CONFIG_DIR=/opt/ml/input/configSM_OUTPUT_DIR=/opt/ml/outputSM_NUM_CPUS=16SM_NUM_GPUS=0SM_MODEL_DIR=/opt/ml/modelSM_MODULE_DIR=s3://sagemaker-us-west-2-876895323997/sagemaker-scikit-learn-2021-07-27-00-42-38-097/source/sourcedir.tar.gzSM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"train":"/opt/ml/input/data/train"},"current_host":"algo-1","framework_module":"sagemaker_sklearn_container.training:main","hosts":["algo-1"],"hyperparameters":{"n-best":10},"input_config_dir":"/opt/ml/input/config","input_data_config":{"train":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"sagemaker-scikit-learn-2021-07-27-00-42-38-097","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-west-2-876895323997/sagemaker-scikit-learn-2021-07-27-00-42-38-097/source/sourcedir.tar.gz","module_name":"train_tfidf","network_interface_name":"eth0","num_cpus":16,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"},"user_entry_point":"train_tfidf.py"}SM_USER_ARGS=["--n-best","10"]SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediateSM_CHANNEL_TRAIN=/opt/ml/input/data/trainSM_HP_N-BEST=10PYTHONPATH=/opt/ml/code:/miniconda3/bin:/miniconda3/lib/python37.zip:/miniconda3/lib/python3.7:/miniconda3/lib/python3.7/lib-dynload:/miniconda3/lib/python3.7/site-packagesInvoking script with the following command:/miniconda3/bin/python train_tfidf.py --n-best 10
Traceback (most recent call last):
  File "train_tfidf.py", line 25, in <module>
    train_table = pd.read_csv(filename)NameError: name 'filename' is not defined2021-07-27 00:45:21,618 sagemaker-containers ERROR    Reporting training FAILURE2021-07-27 00:45:21,619 sagemaker-containers ERROR    framework error: Traceback (most recent call last):
  File "/miniconda3/lib/python3.7/site-packages/sagemaker_containers/_trainer.py", line 84, in train
    entrypoint()
  File "/miniconda3/lib/python3.7/site-packages/sagemaker_sklearn_container/training.py", line 39, in main
    train(environment.Environment())
  File "/miniconda3/lib/python3.7/site-packages/sagemaker_sklearn_container/training.py", line 35, in train
    runner_type=runner.ProcessRunnerType)
  File "/miniconda3/lib/python3.7/site-packages/sagemaker_training/entry_point.py", line 100, in run
    wait, capture_error
  File "/miniconda3/lib/python3.7/site-packages/sagemaker_training/process.py", line 161, in run
    cwd=environment.code_dir,
  File "/miniconda3/lib/python3.7/site-packages/sagemaker_training/process.py", line 81, in check_error
    raise error_class(return_code=return_code, cmd=" ".join(cmd), output=stderr)sagemaker_training.errors.ExecuteUserScriptError: ExecuteUserScriptError:Command "/miniconda3/bin/python train_tfidf.py --n-best 10"ExecuteUserScriptError:Command "/miniconda3/bin/python train_tfidf.py --n-best 10"
```

