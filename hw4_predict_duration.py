import pickle
from pathlib import Path

import click
import pandas as pd


MODELS_DIR = Path("models")
PREDICTIONS_DIR = Path("predicitions")
CAT_FEATURES = ['PUlocationID', 'DOlocationID']


def read_data(url):
    df = pd.read_parquet(url)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CAT_FEATURES] = df[CAT_FEATURES].fillna(-1).astype('int').astype('str')

    return df


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("year", type=click.INT)
@click.argument("month", type=click.INT)
def pipeline(model_path: str, year: int, month: int) -> float:

    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = df[CAT_FEATURES].to_dict(orient='records')

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['predictions'] = y_pred

    output_file = Path(f"predictions/hw4-result-{year}-{month}.parquet")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print(y_pred.mean())


if __name__ == "__main__":
    pipeline()
