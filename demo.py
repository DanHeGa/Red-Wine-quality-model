import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_squared_error, 
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet #regression model
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging

import dagshub
dagshub.init(repo_owner='DanHeGa', repo_name='Red-Wine-quality-model', mlflow=True)


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, prediction):
    rmse = np.sqrt(mean_squared_error(actual, prediction))
    mae = mean_absolute_error(actual, prediction)
    r2 = r2_score(actual, prediction)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    #read the dataset from the csv url
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    try:
        #get data from url
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            f"Invalid or corrupted URL, error: {e}"
        )

    #split into train and test data
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1) #independent variables, drop dependant
    train_y = train[["quality"]] #desired prediction values
    test_x = test.drop(["quality"], axis=1)
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 #learning rate
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 1 else 0.5
    

    with mlflow.start_run(): #automatic experiment tool
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predictions = lr.predict(test_x)

        rmse, mae, r2 = eval_metrics(test_y, predictions)

        #log used parameters and metric values
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        #using Dagshub as the model server
        server_url = "https://dagshub.com/DanHeGa/Red-Wine-quality-model.mlflow"
        mlflow.set_tracking_uri(server_url)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticNetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")






