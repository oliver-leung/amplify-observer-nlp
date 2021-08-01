import boto3
import json
import pandas as pd
import io
from pathlib import Path
from time import time
from progress.bar import Bar

def combine_dfs(dfs):
    """Concatenate a list of DataFrames in the order in which they are listed, then reset
    the index.
    """
    df = pd.concat(dfs, ignore_index=True)
    df = df.reset_index(drop=True)
    return df

def get_corpus_labels(df, corpus_col, label_cols):
    corpus = df[corpus_col]
    label_cols_lst = [df[lab] for lab in label_cols]
    labels = list(zip(*label_cols_lst))
    
    return corpus, labels

def get_secrets():
    secret_name = "SageMakerS3Access"
    region_name = "us-west-2"
    
    secrets = boto3.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    secrets_response = secrets.get_secret_value(SecretId=secret_name)
    secrets_dict = json.loads(secrets_response['SecretString'])
    (access_key, secret_key), = secrets_dict.items()
    return access_key, secret_key

def list_data_objs():
    bucket_name = 'amplifyobserverinsights-aoinsightslandingbucket29-5vcr471d4nm5'
    bucket_subfolder = 'data/issues/'

    s3 = boto3.client('s3')
    data_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=bucket_subfolder)['Contents']
    data_obj_names = [key['Key'] for key in data_objects if key['Key'] != bucket_subfolder]

    return data_obj_names

def download_data(filename, data_obj_names, verbose=False):
    start = time()
    dfs = []
    s3 = boto3.client('s3')
    bucket_name = 'amplifyobserverinsights-aoinsightslandingbucket29-5vcr471d4nm5'

    with Bar(
        message='Downloading parquets',
        check_tty=False,
        hide_cursor=False,
        max=len(data_obj_names)
    ) as bar:

        for obj_name in data_obj_names:
            obj = s3.get_object(Bucket=bucket_name, Key=obj_name)
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            dfs.append(df)
            bar.next()

        bar.finish()

    print('Took', time() - start, 'seconds') if verbose else None
    return dfs

def deserialize_data(filename, verbose=False):
    start = time()
    data = Path(filename)
    
    if data.is_file():
        df = pd.read_csv(filename)
    else:
        raise OSError(filename + ' is not a file or doesn\'t exist.')
        
    print('Deserializing data from', filename, 'took', time() - start, 'seconds') if verbose else None
    return df

def get_data(filename, force_redownload=False, verbose=False):
    data = Path(filename)
    
    if force_redownload or not data.is_file():
        data_obj_names = list_data_objs()
        dfs = download_data(filename, data_obj_names, verbose=verbose)
        df = combine_dfs(dfs)
        df.to_csv(filename, index=False)
    else:
        df = deserialize_data(filename, verbose=verbose)
        
    return df