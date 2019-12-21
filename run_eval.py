from utils.evaluate import Predictor

predictor = Predictor('configs.default')
res = predictor.predict('Hello world!')
print("response:" + res)
