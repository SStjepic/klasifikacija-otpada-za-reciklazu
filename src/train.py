import os
import keras
from keras import layers, models, callbacks, losses, metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input
import config
keras.utils.set_random_seed(42)


datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

datagen_valid = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = datagen_train.flow_from_directory(
    config.TRAIN_PATH,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = datagen_valid.flow_from_directory(
    config.VAL_PATH,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


def create_models_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    create_models_directory(config.MODELS_PATH)
    num_classes = train_data.num_classes
    class_names = list(train_data.class_indices.keys())

    model = MobileNetV3Large(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False
    )
    model.trainable = False 

    inputs = keras.Input(shape=(224, 224, 3))
    x = model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(), 
        metrics=[
            'accuracy', 
            metrics.Precision(name='precision'), 
            metrics.Recall(name='recall')
        ]
    )

    log_path = os.path.join(config.MODELS_PATH, 'training_log.csv')
    model_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=config.BEST_MODEL_PATH, monitor='val_loss', save_best_only=True),
        callbacks.CSVLogger(log_path, separator=",", append=False)
    ]

    history = model.fit(
        train_data,
        epochs=config.EPOCHS,
        validation_data=val_data,
        callbacks=model_callbacks
    )

    model.save(config.FINAL_MODEL_PATH)
    print("\nTreniranje je završeno.")