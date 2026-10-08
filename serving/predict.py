"""Send rows of a CSV to the Seldon deployment and print the predictions."""
import argparse
import json

import pandas as pd
import requests

NON_FEATURE_COLUMNS = ['TIMESTAMP', 'DateTime', 'target']


def model_feature_names(model_uri: str) -> list:
    """Input column names from the MLflow model signature, in training order."""
    import mlflow.pyfunc
    schema = mlflow.pyfunc.load_model(model_uri).metadata.get_input_schema()
    if schema is None:
        raise ValueError(f'{model_uri} has no input signature')
    return schema.input_names()


def build_payload(df: pd.DataFrame, feature_names: list = None) -> dict:
    """Seldon v1 payload. With ``feature_names``, send exactly those columns."""
    if feature_names is not None:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f'Input is missing model features: {missing}')
        features = df[feature_names]
    else:
        features = df.drop(columns=[c for c in df.columns
                                    if c in NON_FEATURE_COLUMNS
                                    or c.startswith('Unnamed:')])
    return {'data': {'names': list(features.columns),
                     'ndarray': features.values.tolist()}}


def parse_response(response: dict) -> list:
    data = response['data']
    if 'ndarray' in data:
        return data['ndarray']
    return data['tensor']['values']


def prediction_url(host: str, namespace: str, deployment_name: str) -> str:
    return (f'{host.rstrip("/")}/seldon/{namespace}/{deployment_name}'
            '/api/v1.0/predictions')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='CSV with the same feature columns used in training')
    parser.add_argument('--host', default='http://localhost:8080',
                        help='Istio ingress gateway URL')
    parser.add_argument('--namespace', default='kubeflow-user-example-com')
    parser.add_argument('--deployment_name', default='washing-machine')
    parser.add_argument('--rows', type=int, default=10)
    parser.add_argument('--cookie', help='authservice_session cookie, if required')
    parser.add_argument('--model_uri',
                        help='MLflow model URI (e.g. models:/WashingMachineModel/Production) '
                             'whose signature selects the columns to send')
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv, nrows=args.rows)
    feature_names = model_feature_names(args.model_uri) if args.model_uri else None
    cookies = {'authservice_session': args.cookie} if args.cookie else None
    response = requests.post(
        prediction_url(args.host, args.namespace, args.deployment_name),
        json=build_payload(df, feature_names), cookies=cookies, timeout=30)
    response.raise_for_status()
    predictions = parse_response(response.json())

    if 'target' in df.columns:
        for expected, predicted in zip(df['target'], predictions):
            print(f'expected={expected} predicted={predicted}')
    else:
        print(json.dumps(predictions))


if __name__ == '__main__':
    main()
