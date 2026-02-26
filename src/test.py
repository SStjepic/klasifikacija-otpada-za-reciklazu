import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes
    class_labels = list(test_data.class_indices.keys())

    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().round(3)
    
    df_report.to_csv(config.MODEL_METRICS, index=True)

    print("\n" + "*"*30)
    print("REZULTATI")
    print("*"*30)
    print(df_report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix - Test Skup')
    plt.ylabel('Stvarna klasa')
    plt.xlabel('Predviđena klasa')
    
    report_path = os.path.join(config.MODELS_PATH, 'test_results.png')
    plt.savefig(report_path)
    print("Confusion matrix je sačuvana.")
    plt.show()
    
    
    df = pd.read_csv(config.TRAIN_LOG) 

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['accuracy'], label='Trening Tačnost', marker='o', linewidth=2)
    plt.plot(df['epoch'], df['val_accuracy'], label='Validaciona Tačnost', marker='o', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoha', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../models/accuracy_plot.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], label='Trening Gubitak', marker='o', linewidth=2, color='red')
    plt.plot(df['epoch'], df['val_loss'], label='Validacioni Gubitak', marker='o', linewidth=2, color='orange')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoha', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../models/loss_plot.png', dpi=300)
    plt.show()