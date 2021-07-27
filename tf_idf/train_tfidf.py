import argparse
import os

import joblib
import pandas as pd
from sklearn import tree
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Set hyperparameters
    parser.add_argument("--n-best", type=int, default=10)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    # Grab args
    args,_ = parser.parse_known_args()
    n_best = args.n_best

    # Select corpus and labels
    filenames = os.listdir(args.train)
    files = [os.path.join(args.train, fn) for fn in filenames]
    dfs = [pd.read_parquet(file, engine='pyarrow') for file in files]
    train_table = pd.concat(
            dfs,
            ignore_index=True
        )

    # Clear empty values and reset indices
    train_table = train_table[(not isinstance(df.bodyText, str)) and (df.bodyText != '')]
    train_table = train_table.reset_index(drop=True)
    train_X = train_table['bodyText']
    train_y = train_table['url']
    
    # Create and train pipeline
    clf = make_pipeline(
        TfidfVectorizer(
            tokenizer=LemmaTokenizer(custom=True)
        ),
        VectorSimilarity(
            n_best=n_best
        )
    )
    clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
