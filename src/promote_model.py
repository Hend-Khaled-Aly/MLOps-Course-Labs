from mlflow.tracking import MlflowClient
import mlflow

# Set the tracking URI to your MLflow tracking server location
mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

client.transition_model_version_stage(
    name="RandomForest",
    version=2,
    stage="Production",
    archive_existing_versions=True
)

print("Model version 2 promoted to Production successfully.")
