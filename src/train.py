import os
from src.preprocess import load_data
from src.model import build_model

def train_model():
    (x_train , y_train) , (x_test , y_test) = load_data()
    model = build_model()

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    history = model.fit(
        x_train , y_train,
        epochs = 5,
        batch_size = 64 , 
        validation_split = 0.1
    )

    os.makedirs('model' , exist_ok=True)
    model.save('model/digit_recognizer.keras')
    print("MODEL SAVED!!")
    return model , history