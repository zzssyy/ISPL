from sklearn import metrics
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale

def identify(datas1, datas2, m, n):
    y = np.ones(m).tolist() + np.zeros(n).tolist()
    X = np.concatenate((np.array(datas1)[1:, 1:], np.array(datas2)[1:, 1:]), axis=1).tolist()
    X = minmax_scale(X)
    model = joblib.load("train_model.m")
    pred = model.predict(X)
    print(metrics.confusion_matrix(y, pred))

def predict(datas1, datas2):
    X = np.concatenate((np.array(datas1)[1:, 1:], np.array(datas2)[1:, 1:]), axis=1).tolist()
    X = minmax_scale(X)
    model = joblib.load("train_model.m")
    pred = model.predict(X)
    print(pred)
