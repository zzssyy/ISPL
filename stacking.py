from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.externals import joblib

def data_load(path1, path2):
    datas1 = list()   
    with open(path1, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas1.append(s[1:])
    labels = np.ones(int(len(datas1)/2)).tolist() + np.zeros(int(len(datas1)/2)).tolist()
    len1 = len(datas1[0])
    datas2 = list()
    with open(path2, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas2.append(s[1:])
    len2 = len(datas2[0])
    datas = np.concatenate((np.array(datas1[1:]), np.array(datas2[1:])), axis=1).tolist()
    print(len1)
    print(len2)

    return datas, labels, len1, len2

def split_data(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    return x_train, x_test, y_train, y_test

def model(x_train, x_test, y_train, y_test, len1, len2):
    clf = RandomForestClassifier(random_state=1, n_estimators=100)  
    pipe1 = make_pipeline(ColumnSelector(cols=range(0, len1)), clf)
    pipe2 = make_pipeline(ColumnSelector(cols=range(len1, len1+len2)), clf)
	
    sclf = StackingCVClassifier(classifiers=[pipe1, pipe2], 
							  meta_classifier=LogisticRegression())
	
    sclf.fit(np.array(x_train), y_train)
    pred_testlabel = sclf.predict(np.array(x_test))
    print('准确率', metrics.accuracy_score(y_test, pred_testlabel))
    print('召回率', metrics.recall_score(y_test, pred_testlabel,average='weighted'))
    print('精确率', metrics.precision_score(y_test, pred_testlabel,average='weighted'))
    print('f1-score:', metrics.f1_score(y_test, pred_testlabel,average='weighted'))
    
    joblib.dump(sclf, "train_model.m")

path1 = ''
path2 = ''
X, y, len1, len2 = data_load(path1, path2)
x_train, x_test, y_train, y_test = split_data(X, y)
model(x_train, x_test, y_train, y_test, len1, len2)