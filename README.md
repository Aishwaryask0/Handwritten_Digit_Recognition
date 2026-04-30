# Handwritten Digit Recognition CNN

A Convolutional Neural Network (CNN) built with TensorFlow and Keras 
that classifies handwritten digits with 99%+ accuracy.

## Features
- CNN model trained on 60,000 MNIST images
- 99%+ accuracy on unseen test data
- Confusion matrix visualization
- Real image prediction (draw a digit → model predicts!)

## Technologies Used
- Python 3.12
- TensorFlow 2.21
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## Model Architecture
Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output(10)

## Results
- Training Accuracy: 99.3%
- Test Accuracy: 99.17%

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`
4. Predict your own image: `python -c "from src.predict import predict_real_image; predict_real_image('your_image.png')"`