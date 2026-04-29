import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from src.preprocess import load_data
from tensorflow.keras.models import load_model

def evaluate_model():
    (x_train , y_train) , (x_test , y_test) = load_data()
    model = load_model('model/digit_recognizer.keras')

    test_loss , test_accuracy  = model.evaluate(x_test , y_test)

    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"Test Loss : {test_loss:.4f}")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred , axis= 1)

    cm = confusion_matrix( y_test , y_pred_classes)
    disp = ConfusionMatrixDisplay( confusion_matrix= cm , display_labels=list(range(10)))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('images/confusion_matrix.png')
    plt.show()
    print("CONFUSION MATRIX SAVED!!!")