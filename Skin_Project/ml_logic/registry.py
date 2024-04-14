import time
import os
import glob

from tensorflow import keras
import mlflow
from Skin_Project.params import *
from mlflow.tracking import MlflowClient
from keras.models import load_model

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(CHEMIN_4, f"{timestamp}.h5")
    model.save(model_path)
    print("Model saved locally")

    if MODEL_TARGET == "mlflow":
        if CLASSIFICATION=="binary":
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="tmp/mlartifacts",
                registered_model_name="skin project binary model"
            )

            print("✅ Binary model saved to MLflow")

        elif CLASSIFICATION=='cat':
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="tmp/mlartifacts",
                registered_model_name="skin project categorical model"
            )

            print("✅ Categorical model saved to MLflow")

        else :
            print("❌ Please select a classification 'binary' or 'cat'")
        return None

    return None



def load_best_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """
    if METADATA == 'no':
        if MODEL_TARGET == "local":

            if CLASSIFICATION == 'binary':
                # Get the latest model version name by the timestamp on disk
                local_best_model_path = f"{CHEMIN_BINARY}/best_model.h5"

                if not local_best_model_path:
                    print('No best model saved yet')
                    return None

                best_model = keras.saving.load_model(local_best_model_path)

                print("Best model for binary classification loaded from local disk")

                return best_model

            if CLASSIFICATION == 'cat':
                # Get the latest model version name by the timestamp on disk
                local_best_model_path = f"{CHEMIN_CAT}/best_model.h5"

                if not local_best_model_path:
                    print('No best model saved yet')
                    return None

                best_model = keras.saving.load_model(local_best_model_path)

                print("Best model for multiclass classification loaded from local disk")

                return best_model
        pass

    elif METADATA == 'yes':
        if MODEL_TARGET == "local":

            if CLASSIFICATION == 'binary':
                # Get the latest model version name by the timestamp on disk
                local_best_model_path = f"{CHEMIN_META_BINARY}/best_model.h5"

                if not local_best_model_path:
                    print('No best model saved yet')
                    return None, None

                best_model = keras.saving.load_model(local_best_model_path)

                print("Best model cnn for binary classification loaded from local disk")

                local_best_model_ml_path = f"{CHEMIN_META_BINARY}/best_model_ml.h5"
                best_model_ml = keras.saving.load_model(local_best_model_ml_path)

                print("Best model ml for binary classification loaded from local disk")

                return best_model, best_model_ml

            if CLASSIFICATION == 'cat':
                # Get the latest model version name by the timestamp on disk
                local_best_model_path = f"{CHEMIN_META_CAT}/best_model.h5"

                if not local_best_model_path:
                    print('No best model saved yet')
                    return None, None

                best_model = keras.saving.load_model(local_best_model_path)

                print("Best model for multiclass classification loaded from local disk")

                local_best_model_ml_path = f"{CHEMIN_META_CAT}/best_model_ml.h5"
                best_model_ml = keras.saving.load_model(local_best_model_ml_path)

                print("Best model ml for multiclass classification loaded from local disk")

                return best_model, best_model_ml
        pass


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{CHEMIN_4}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print('local_model_paths',local_model_paths)

        latest_model = keras.saving.load_model(most_recent_model_path_on_disk)

        print("Model loaded from local disk")

        return latest_model



    elif MODEL_TARGET == "mlflow":
        # load model from ML Flow

        if CLASSIFICATION=="binary":
            model = None
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            try:
                model_versions = client.get_latest_versions(name="skin project binary model", stages=[stage])
                model_uri = model_versions[0].source

                assert model_uri is not None

            except:
                print(f"\n❌ No model found with name 'skin project binary model' in stage {stage}")

                return None

            model = mlflow.tensorflow.load_model(model_uri=model_uri)

            print("✅ Model binary loaded from MLflow")
            return model

        elif CLASSIFICATION=="cat":
            model = None
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            try:
                model_versions = client.get_latest_versions(name="skin project categorical model", stages=[stage])
                model_uri = model_versions[0].source

                assert model_uri is not None

            except:
                print(f"\n❌ No model found with name 'skin project categorical model' in stage {stage}")

                return None

            model = mlflow.tensorflow.load_model(model_uri=model_uri)

            print("✅ Model categorical loaded from MLflow")
            return model
    else:
        return None


def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    if CLASSIFICATION=='binary':
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = MlflowClient()

        version = client.get_latest_versions(name='skin project binary model', stages=[current_stage])

        if not version:
            print(f"\n❌ No model found with name 'skin project binary model' in stage {current_stage}")
            return None

        client.transition_model_version_stage(
            name='skin project binary model',
            version=version[0].version,
            stage=new_stage,
            archive_existing_versions=True
        )

        print(f"✅ Model 'skin project binary model' (version {version[0].version}) transitioned from {current_stage} to {new_stage}")
        return None

    elif CLASSIFICATION=='cat':
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = MlflowClient()

        version = client.get_latest_versions(name='skin project categorical model', stages=[current_stage])

        if not version:
            print(f"\n❌ No model found with name 'skin project categorical model' in stage {current_stage}")
            return None

        client.transition_model_version_stage(
            name='skin project categorical model',
            version=version[0].version,
            stage=new_stage,
            archive_existing_versions=True
        )

        print(f"✅ Model 'skin project categorical model' (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

        return None



    return None
