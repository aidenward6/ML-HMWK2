"""
hw2.py homework2
Aiden Ward and Grady Algire, both of us put equal effort into assignment
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iris_utils import *


# part 1: binary logistic regression

def sigmoid(z):
    # clip z so exp doesn't blow up with huge numbers
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))


def binary_loss(theta, Xb, y, lam):
    # cross entropy loss with l2, skip regularizing bias
    n = len(y)
    p = sigmoid(Xb @ theta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ce = - (1 / n) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    reg = (lam / 2) * np.sum(theta[1:] ** 2)
    return ce + reg


def binary_grad(theta, Xb, y, lam):
    # gradient formula from eq 2
    n = len(y)
    p = sigmoid(Xb @ theta)
    theta_reg = theta.copy()
    theta_reg[0] = 0
    grad = (1 / n) * (Xb.T @ (p - y)) + lam * theta_reg
    return grad


class BinaryLogisticRegression:
    def __init__(self, eta=0.5, lam=0.01, max_iter=2000):
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        # add bias column of 1s and run batch gd
        Xb = add_bias(X)
        self.theta = np.zeros(Xb.shape[1])
        self.loss_history = []

        for i in range(self.max_iter):
            loss = binary_loss(self.theta, Xb, y, self.lam)
            self.loss_history.append(loss)
            grad = binary_grad(self.theta, Xb, y, self.lam)
            self.theta -= self.eta * grad
        return self

    def predict_proba(self, X):
        Xb = add_bias(X)
        return sigmoid(Xb @ self.theta)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)


# part 2: two-stage classifier and softmax

class TwoStageClassifier:
    def __init__(self, eta=0.5, lam=0.01, max_iter=2000):
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.stage1 = None
        self.stage2 = None

    def fit(self, X, y):
        # stage 1: setosa (1) vs rest (0) on all rows
        y1 = (y == 0).astype(int)
        self.stage1 = BinaryLogisticRegression(self.eta, self.lam, self.max_iter)
        self.stage1.fit(X, y1)

        # stage 2: only train on non-setosa samples
        mask = (y != 0)
        X_sub = X[mask]
        y_sub = y[mask]

        # in non-setosa rows, virginica is 1 and versicolor is 0
        y2 = (y_sub == 2).astype(int)
        self.stage2 = BinaryLogisticRegression(self.eta, self.lam, self.max_iter)
        self.stage2.fit(X_sub, y2)
        return self

    def predict_proba(self, X):
        q1 = self.stage1.predict_proba(X)
        q2 = self.stage2.predict_proba(X)

        # chain rule formula from eq 3
        p_setosa = q1
        p_versicolor = (1 - q1) * (1 - q2)
        p_virginica = (1 - q1) * q2

        probs = np.column_stack([p_setosa, p_versicolor, p_virginica])
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def softmax(S):
    # subtract row max so exp doesn't overflow
    shift_S = S - np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(shift_S)
    return exp_S / np.sum(exp_S, axis=1, keepdims=True)


def softmax_loss(W, Xb, Y, lam):
    # multinomial loss, first row of W is bias so skip it in reg
    n = len(Y)
    P = softmax(Xb @ W)
    P = np.clip(P, 1e-12, 1.0)
    ce = - (1 / n) * np.sum(Y * np.log(P))
    reg = (lam / 2) * np.sum(W[1:, :] ** 2)
    return ce + reg


def softmax_grad(W, Xb, Y, lam):
    # gradient from eq 5, zero out bias row for regularization
    n = len(Y)
    P = softmax(Xb @ W)
    W_reg = W.copy()
    W_reg[0, :] = 0
    grad = (1 / n) * (Xb.T @ (P - Y)) + lam * W_reg
    return grad


class SoftmaxRegression:
    def __init__(self, eta=0.5, lam=0.01, max_iter=3000):
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.W = None
        self.loss_history = []

    def fit(self, X, y):
        Xb = add_bias(X)
        Y = one_hot(y, 3)
        self.W = np.zeros((Xb.shape[1], 3))
        self.loss_history = []

        for i in range(self.max_iter):
            loss = softmax_loss(self.W, Xb, Y, self.lam)
            self.loss_history.append(loss)
            grad = softmax_grad(self.W, Xb, Y, self.lam)
            self.W -= self.eta * grad
        return self

    def predict_proba(self, X):
        Xb = add_bias(X)
        return softmax(Xb @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == "__main__":
    # load and split data with seed 0
    X, y = load_iris_csv("iris.csv")
    Xtr_raw, Xte_raw, ytr, yte = stratified_split(X, y, test_fraction=0.3, seed=0)
    Xtr, Xte, mu, sd = standardize(Xtr_raw, Xte_raw)

    print("running homework 2 models...")

    # --- part 1.2: setosa vs rest ---
    print("\npart 1.2: binary logistic regression (setosa vs rest)")
    ytr_setosa = (ytr == 0).astype(int)
    yte_setosa = (yte == 0).astype(int)

    binary_model = BinaryLogisticRegression(eta=0.5, lam=0.1, max_iter=2000)
    binary_model.fit(Xtr, ytr_setosa)

    train_acc = accuracy(ytr_setosa, binary_model.predict(Xtr))
    test_acc = accuracy(yte_setosa, binary_model.predict(Xte))
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy: {test_acc:.4f}")

    # plot loss curve for binary model
    plt.figure(figsize=(6, 4))
    plt.plot(binary_model.loss_history)
    plt.title("Binary Logistic Regression: Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig1_binary_loss.png")
    plt.close()
    print("saved fig1_binary_loss.png")

    # --- part 2.3: comparing two-stage vs softmax ---
    print("\npart 2.3: two-stage vs softmax")

    # two stage
    two_stage = TwoStageClassifier(eta=0.5, lam=0.01, max_iter=2000)
    two_stage.fit(Xtr, ytr)
    ts_train_acc = accuracy(ytr, two_stage.predict(Xtr))
    ts_test_acc = accuracy(yte, two_stage.predict(Xte))
    ts_cm = confusion_matrix(yte, two_stage.predict(Xte))

    print("two-stage classifier:")
    print(f"  train accuracy: {ts_train_acc:.4f}")
    print(f"  test accuracy: {ts_test_acc:.4f}")
    print("  confusion matrix (rows=true, cols=pred):")
    print(ts_cm)

    # softmax
    softmax_model = SoftmaxRegression(eta=0.5, lam=0.01, max_iter=3000)
    softmax_model.fit(Xtr, ytr)
    sm_train_acc = accuracy(ytr, softmax_model.predict(Xtr))
    sm_test_acc = accuracy(yte, softmax_model.predict(Xte))
    sm_cm = confusion_matrix(yte, softmax_model.predict(Xte))

    print("\nsoftmax regression:")
    print(f"  train accuracy: {sm_train_acc:.4f}")
    print(f"  test accuracy: {sm_test_acc:.4f}")
    print("  confusion matrix (rows=true, cols=pred):")
    print(sm_cm)

    # scatter plot of petal length vs petal width
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(Xtr_raw[ytr == 0, 2], Xtr_raw[ytr == 0, 3], color="green", label="setosa")
    plt.scatter(Xtr_raw[ytr == 1, 2], Xtr_raw[ytr == 1, 3], color="blue", label="versicolor")
    plt.scatter(Xtr_raw[ytr == 2, 2], Xtr_raw[ytr == 2, 3], color="red", label="virginica")
    plt.title("Petal Length vs Petal Width (Training Set)")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig2_petal_scatter.png")
    plt.close()
    print("saved fig2_petal_scatter.png")

    print("\ndone!")
