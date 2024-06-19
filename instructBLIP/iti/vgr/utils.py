import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def train_single_prob(X_all, y_all, val_size):
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=val_size)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    return clf, val_acc
