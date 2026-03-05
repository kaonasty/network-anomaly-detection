import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
from mlflow.tracking import MlflowClient
client = MlflowClient()
for mv in client.search_model_versions("name='network-anomaly-detector'"):
    print(f'Version: {mv.version}, Stage: {mv.current_stage}, Alias: {mv.aliases}')