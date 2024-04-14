import os

IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE'))
CHEMIN_1 = os.environ.get('CHEMIN_1')
CHEMIN_2 = os.environ.get('CHEMIN_2')
CHEMIN_3 = os.environ.get('CHEMIN_3')
CHEMIN_4 = os.environ.get('CHEMIN_4')
CHEMIN_BINARY = os.environ.get('CHEMIN_BINARY')
CHEMIN_CAT = os.environ.get('CHEMIN_CAT')
CHEMIN_META_BINARY = os.environ.get('CHEMIN_META_BINARY')
CHEMIN_META_CAT = os.environ.get('CHEMIN_META_CAT')
CHEMIN_METADATA = os.environ.get('CHEMIN_METADATA')
CHEMIN_TEST = os.environ.get('CHEMIN_TEST')
MODEL_TARGET = os.environ.get("MODEL_TARGET")
THRESHOLD = float(os.environ.get('THRESHOLD'))
CLASSIFICATION = str(os.environ.get("CLASSIFICATION"))
SAMPLE_SIZE = float(os.environ.get('SAMPLE_SIZE',0.5))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")

API_URL = os.environ.get("API_URL")
IMAGE_DIR = os.environ.get("IMAGE_DIR")

METADATA=os.environ.get("METADATA")
