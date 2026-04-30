import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.preprocess import load_data
from PIL import Image

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

def predict_real_image(image_path):
    img =  Image.open(image_path).convert('L')
    img = img.resize((28,28))
    
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1,28,28,1)

    model = load_model('model/digit_recognizer.keras')
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    plt.imshow(img_array.reshape(28,28) , cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
    print(f"Predicted Digit : {predicted_label}")
