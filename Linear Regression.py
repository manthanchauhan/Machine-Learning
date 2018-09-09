import numpy as np
from sklearn import model_selection

def fit(x_train, y_train):
    num = (x_train*y_train).mean() - (x_train.mean()*y_train.mean())
    den = (x_train**2).mean() - x_train.mean()**2
    m = num/den
    m *= -1
    c = y_train.mean() - m*x_train.mean()
    return m, c
def predict(m, c, x_test):
    y_pred = [m*x + c for x in x_test]
    return y_pred
def score(y_true,y_pred):
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1-u/v
data = np.loadtxt('data.csv', delimiter=',')
x = data[:, 0]
y = data[:, 1]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
m, c = fit(x_train, y_train)
y_pred = predict(m, c, x_test)
s = score(y_test, y_pred)
pritn(s)

