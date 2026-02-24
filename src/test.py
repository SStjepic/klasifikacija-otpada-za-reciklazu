import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v3 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import config    

if __name__ == '__main__': 
    model = keras.models.load_model(config.BEST_MODEL_PATH)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_data = test_datagen.flow_from_directory(
        config.TEST_PATH,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("Pokretanje evaluacije na test skupu...")
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes
    class_labels = list(test_data.class_indices.keys())

    print("\n" + "*"*30)
    print("IZVEŠTAJ KLASIFIKACIJE")
    print("*"*30)
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix - Test Skup')
    plt.ylabel('Stvarna klasa')
    plt.xlabel('Predviđena klasa')
    
    report_path = '../models/test_results.png'
    plt.savefig(report_path)
    print(f"\nMatrica konfuzije je sačuvana.")
    plt.show()