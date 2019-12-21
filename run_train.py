from utils.train import train
from utils.evaluate import Predictor

train('configs.default')
predictor = Predictor('configs.default')
res = predictor.predict('Hello world!')
print("response:" + res)
