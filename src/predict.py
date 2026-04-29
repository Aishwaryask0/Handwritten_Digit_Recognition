import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.preprocess import load_data

def predict_single():
    (X_train , y_train) , (x_test , y_test) = load_data()
    model = load_model('model/digit_recognizer.keras')

    index = np.random.randint(0 , len(x_test))
    image = x_test[index]
    true_label = y_test[index]

    prediction = model.predict(image.reshape(1,28,28,1))
    predicted_label = np.argmax(prediction)

    plt.imshow(image.reshape(28,28) , cmap = 'grey')
    plt.title(f"True : {true_label} | Prediction : {predicted_label}")
    plt.axis('off')
    plt.savefig('images/sample_prediction.png')
    plt.show()
    print(f"True Label : {true_label}")
    print(f"prediction :  {predicted_label}")
