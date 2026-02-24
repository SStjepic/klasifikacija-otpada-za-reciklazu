IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30


DATA_PATH = '../data/processed'
TRAIN_PATH = f"{DATA_PATH}/train"
VAL_PATH = f"{DATA_PATH}/val"
TEST_PATH = f"{DATA_PATH}/test"

MODELS_PATH = '../models'
BEST_MODEL_PATH = f"{MODELS_PATH}/best_model.keras"
FINAL_MODEL_PATH = f"{MODELS_PATH}/model.keras"