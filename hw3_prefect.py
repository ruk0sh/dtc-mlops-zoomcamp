import datetime
import pickle
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Tuple, Union

import click
import pandas as pd
import prefect.logging
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task()
def get_paths(
    date: Union[datetime.date, None] = None,
    train_offset_months: int = -2,
    val_offset_months: int = -1,
    fhv_data_dir: Path = Path("D:\\data\\nyc-taxi\\fhv"),
) -> Tuple[Path, Path, datetime.date]:

    if not date:
        date = datetime.date.today()
    train_date = date + relativedelta(months=train_offset_months)
    train_path = fhv_data_dir / f"fhv_tripdata_{train_date.year}-{train_date.month:02d}.parquet"
    val_date = date + relativedelta(months=val_offset_months)
    val_path = fhv_data_dir / f"fhv_tripdata_{val_date.year}-{val_date.month:02d}.parquet"
    return train_path, val_path, date


@task()
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task()
def prepare_features(df, categorical, train=True):
    logger = prefect.logging.get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task()
def train_model(df, categorical):
    logger = prefect.logging.get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task()
def run_model(df, categorical, dv, lr):
    logger = prefect.logging.get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")


@click.command()
@click.argument("date", type=click.DateTime(), default="2021-08-15")
def cli(date: datetime.date):
    main_flow(date)


@flow(task_runner=SequentialTaskRunner())
def main_flow(date=None):
    train_path, val_path, date_resolved = get_paths(date=date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path).result()
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path).result()
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # save model
    save_model_path = Path(f"models/model-{str(date_resolved)[:10]}.bin")
    with open(save_model_path, "wb") as lr_file:
        pickle.dump(lr, lr_file)
    save_vectorizer_path = Path(f"models/dv-{str(date_resolved)[:10]}.bin")
    with open(save_vectorizer_path, "wb") as dv_file:
        pickle.dump(dv, dv_file)

    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    flow=main_flow,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
)


if __name__ == "__main__":
    cli()
