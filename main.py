from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_single

print("------HANDWRITTEN DIGIT RECOGNITION------")
print("1.TRAINING MODEL")
train_model()
print("2.EVALUATE MODEL ")
evaluate_model()
print("3.PRIDICT MODEL")
predict_single()
print("ALL DONE!!!")