import os
import pickle
import click
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Set our tracking server uri for logging
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create a new MLflow Experiment
mlflow.set_experiment("Green_Taxi_rides")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):
    
    # Start an MLflow run
    with mlflow.start_run():

        # Set a tag to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic RF model for Taxi data")

        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        params={"max_depth":10, "random_state":0}
        
        #Log the hyperparameters
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # Infer the model signature
        signature = infer_signature(X_train, rf.predict(X_train))

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # Log the loss metric
        mlflow.log_metric("rmse", rmse)

        # Log the model
        model_info = mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="RF_green_taxi_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-rf",
    )
        print("Model logged to MLflow!")
    # Retrieve the full model final parameters
        Full_params = rf.get_params()
        print(Full_params)

if __name__ == '__main__':
    run_train()

    