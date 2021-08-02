import argparse
import os

import joblib
import pandas as pd
import json
from tfidf_predictor import TfidfPredictor
from utils_tfidf import combine_dfs


def parse_args():
    global args
    parser = argparse.ArgumentParser()

    # Set hyperparameters
    parser.add_argument("--n-best", type=int, default=10)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()
    return args


def load_files(direc):
    filenames = os.listdir(direc)
    files = [os.path.join(direc, fn) for fn in filenames]
    dfs = []
    for file in files:

        _, ext = os.path.splitext(file)
        if ext == '.parquet':
            dfs.append(pd.read_parquet(file, engine='pyarrow'))
        elif ext == '.csv':
            dfs.append(pd.read_csv9(file))

    df = combine_dfs(dfs)
    return df


if __name__ == "__main__":
    args = parse_args()
    hyperparams = {
        'n_best': args.n_best
    }

    # Select corpus and labels
    train_df = load_files(args.train)
    train_X = train_df['title_body']
    train_y = list(zip(train_df['url'], train_df['title']))

    # Create and train pipeline
    clf = TfidfPredictor(**hyperparams)
    clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def input_fn(request_body, request_content_type):
    print(request_body, request_content_type)

    train_inputs = []
    if request_content_type == 'application/json':
        request = json.loads(request_body)
        train_inputs = request['data']

    elif request_content_type == 'application/octet-stream':
        request = request_body.decode('utf-8')

    request = json.loads(request_body)
    train_inputs = request['data']

    return train_inputs


def predict_fn(input_data, model):
    return model.predict(input_data)

# def output_fn(prediction, content_type):
#     print(prediction, content_type)
#     if content_type == 'application/json':
#         pred = prediction[0][0]
#         pred = '\n'.join(pred)
#         score = prediction[0][1]
#         score = '\n'.join(score)
#         output = 'Hello, I have found some issues that might be similar to yours:\n'
#         output +=pred + '\n'
#         output += 'This is how confident I am in these predictions:\n'
#         output += score
#         return output
