import numpy as np
import keras
from keras.applications.mobilenet_v3 import preprocess_input
from keras.utils import load_img, img_to_array
import config
import os

    
if __name__ == "__main__":
    model = keras.models.load_model(config.BEST_MODEL_PATH)

    class_labels = sorted(os.listdir(config.TRAIN_PATH))

    while True:
        print("\n" + "="*40)
        img_path = input("Unesite putanju do slike (ili 'exit' za izlaz): ").strip()
        
        if img_path.lower() == 'exit':
            break

        if not os.path.exists(img_path):
            print("Fajl ne postoji.")
            continue

        try:
            img = load_img(img_path, target_size=config.IMG_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array, verbose=0)
            score = float(np.max(predictions))
            class_idx = np.argmax(predictions)
            result = class_labels[class_idx]

            print("-" * 40)
            print(f"PREDIKCIJA: {result.upper()}")
            print(f"POUZDANOST: {score * 100:.2f}%")
            print("-" * 40)

        except Exception as e:
            print("Greška.")
