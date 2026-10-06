"""
hw2.py homework2
Aiden Ward and Grady Algire, both of us put equal effort into assignment
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so plotting doesnt pop up gui
import matplotlib.pyplot as plt

from iris_utils import (
    load_iris_csv,
    stratified_split,
    standardize,
    add_bias,
    one_hot,
    accuracy,
    confusion_matrix,
    CLASS_NAMES,
    FEATURE_NAMES,
)


# part 1: binary logistic regression

def sigmoid(z):
    # stable sigmoid, clip to avoid overflow
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def binary_loss(theta, Xb, y, lam):
    # binary cross-entropy loss with l2, dont regularize bias
    n = len(y)
    p = sigmoid(Xb @ theta)
    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    cross_entropy = - (1.0 / n) * np.sum(
        y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)
    )
    reg_loss = 0.5 * lam * np.sum(theta[1:] ** 2)
    return float(cross_entropy + reg_loss)


def binary_grad(theta, Xb, y, lam):
    # gradient for binary loss, zero out bias penalty
    n = len(y)
    p = sigmoid(Xb @ theta)
    theta_tilde = theta.copy()
    theta_tilde[0] = 0.0
    grad = (1.0 / n) * (Xb.T @ (p - y)) + lam * theta_tilde
    return grad


class BinaryLogisticRegression:
    # basic binary logreg with batch gd
    def __init__(self, eta=0.5, lam=0.01, max_iter=2000):
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        # prepend bias and run batch gd
        Xb = add_bias(X)
        n, d_plus_1 = Xb.shape
        self.theta = np.zeros(d_plus_1)
        self.loss_history = []

        for _ in range(self.max_iter):
            loss = binary_loss(self.theta, Xb, y, self.lam)
            self.loss_history.append(loss)
            grad = binary_grad(self.theta, Xb, y, self.lam)
            self.theta -= self.eta * grad

        return self

    def predict_proba(self, X):
        # return predicted probability p(y=1|x)
        Xb = add_bias(X)
        return sigmoid(Xb @ self.theta)

    def predict(self, X, threshold=0.5):
        # threshold at 0.5
        return (self.predict_proba(X) >= threshold).astype(int)








# part 2: two-stage classifier and softmax

class TwoStageClassifier:
    # chains stage 1 (setosa vs rest) and stage 2 (versicolor vs virginica)
    def __init__(self, eta=0.5, lam=0.01, max_iter=2000):
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.stage1 = None
        self.stage2 = None

    def fit(self, X, y):
        # stage 1: setosa (1) vs rest (0) on everything
        y_stage1 = (y == 0).astype(int)
        self.stage1 = BinaryLogisticRegression(
            eta=self.eta, lam=self.lam, max_iter=self.max_iter
        )
        self.stage1.fit(X, y_stage1)

        # stage 2: only train on non-setosa rows
        non_setosa_mask = (y != 0)
        X_stage2 = X[non_setosa_mask]
        y_stage2_raw = y[non_setosa_mask]

        # virginica is 1, versicolor is 0
        y_stage2 = (y_stage2_raw == 2).astype(int)
        self.stage2 = BinaryLogisticRegression(
            eta=self.eta, lam=self.lam, max_iter=self.max_iter
        )
        self.stage2.fit(X_stage2, y_stage2)
        return self

    def predict_proba(self, X):
        # combine probabilities using the chain rule from eq 3
        q1 = self.stage1.predict_proba(X)
        q2 = self.stage2.predict_proba(X)

        p_setosa = q1
        p_versicolor = (1.0 - q1) * (1.0 - q2)
        p_virginica = (1.0 - q1) * q2

        P = np.column_stack([p_setosa, p_versicolor, p_virginica])
        assert np.allclose(P.sum(axis=1), 1.0), "probabilities must sum to 1"
        return P

    def predict(self, X):
        # pick class with highest prob
        return np.argmax(self.predict_proba(X), axis=1)


def softmax(S):
    # stable softmax, subtract row max so exp doesnt blow up
    S_shift = S - np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S_shift)
    return exp_S / np.sum(exp_S, axis=1, keepdims=True)


def softmax_loss(W, Xb, Y, lam):
    # multinomial loss with l2, dont penalize bias row
    n = len(Y)
    P = softmax(Xb @ W)
    P_clipped = np.clip(P, 1e-12, 1.0)
    cross_entropy = - (1.0 / n) * np.sum(Y * np.log(P_clipped))
    reg_loss = 0.5 * lam * np.sum(W[1:, :] ** 2)
    return float(cross_entropy + reg_loss)


def softmax_grad(W, Xb, Y, lam):
    # softmax gradient, zero out bias row for reg
    n = len(Y)
    P = softmax(Xb @ W)
    W_tilde = W.copy()
    W_tilde[0, :] = 0.0
    grad = (1.0 / n) * (Xb.T @ (P - Y)) + lam * W_tilde
    return grad


class SoftmaxRegression:
    # multiclass softmax with batch gd
    def __init__(self, eta=0.5, lam=0.01, max_iter=3000):
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.W = None
        self.loss_history = []

    def fit(self, X, y, K=3):
        # one hot labels and run gd
        Xb = add_bias(X)
        n, d_plus_1 = Xb.shape
        Y = one_hot(y, K)
        self.W = np.zeros((d_plus_1, K))
        self.loss_history = []

        for _ in range(self.max_iter):
            loss = softmax_loss(self.W, Xb, Y, self.lam)
            self.loss_history.append(loss)
            grad = softmax_grad(self.W, Xb, Y, self.lam)
            self.W -= self.eta * grad

        return self

    def predict_proba(self, X):
        # return class probabilities
        Xb = add_bias(X)
        return softmax(Xb @ self.W)

    def predict(self, X):
        # pick class with highest prob
        return np.argmax(self.predict_proba(X), axis=1)

# helper for printing evaluation stats

def print_evaluation(name, y_true, y_pred, C):
    acc = accuracy(y_true, y_pred)
    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f} ({int(round(acc * len(y_true)))}/{len(y_true)})")
    print("Confusion Matrix (Rows = True Class [0: setosa, 1: versicolor, 2: virginica], Cols = Predicted):")
    print(C)
    print("-" * 55)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    precisions, recalls, f1s = [], [], []
    for k, cname in enumerate(CLASS_NAMES):
        col_sum = np.sum(C[:, k])
        row_sum = np.sum(C[k, :])
        prec = C[k, k] / col_sum if col_sum > 0 else 0.0
        rec = C[k, k] / row_sum if row_sum > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        print(f"{cname:<12} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
    print("-" * 55)
    print(f"Macro Precision: {np.mean(precisions):.4f}")
    print(f"Macro Recall:    {np.mean(recalls):.4f}")
    print(f"Macro F1-Score:  {np.mean(f1s):.4f}\n")









# main execution pipeline

def main():
    print("=" * 70)
    print("Programming Assignment #2: Two Binary Classifiers vs. Softmax")
    print("=" * 70)

    # 1. load data and make reproducible split (seed 0)
    seed = 0
    X, y = load_iris_csv("iris.csv")
    Xtr_raw, Xte_raw, ytr, yte = stratified_split(X, y, test_fraction=0.3, seed=seed)
    Xtr, Xte, mu, sd = standardize(Xtr_raw, Xte_raw)

    print(f"Dataset split (seed={seed}):")
    print(f"  Training samples: {len(ytr)} (per class: {np.bincount(ytr)})")
    print(f"  Testing samples:  {len(yte)} (per class: {np.bincount(yte)})")
    print(f"  Feature count:    {Xtr.shape[1]}")
    print("-" * 70)

    # part 1.2: train binary classifier (setosa vs rest) with lam = 0.1
    print("\n--- Part 1.2: Binary Logistic Regression (Setosa vs. Rest) ---")
    ytr_setosa = (ytr == 0).astype(int)
    yte_setosa = (yte == 0).astype(int)

    binary_clf = BinaryLogisticRegression(eta=0.5, lam=0.1, max_iter=2000)
    binary_clf.fit(Xtr, ytr_setosa)

    tr_acc_bin = accuracy(ytr_setosa, binary_clf.predict(Xtr))
    te_acc_bin = accuracy(yte_setosa, binary_clf.predict(Xte))
    print(f"Train Accuracy: {tr_acc_bin:.4f} ({int(round(tr_acc_bin * len(ytr_setosa)))}/{len(ytr_setosa)})")
    print(f"Test Accuracy:  {te_acc_bin:.4f} ({int(round(te_acc_bin * len(yte_setosa)))}/{len(yte_setosa)})")
    print(f"Initial Loss J(theta_0): {binary_clf.loss_history[0]:.6f}")
    print(f"Final Loss J(theta_2000): {binary_clf.loss_history[-1]:.6f}")

    # save fig 1: binary loss curve
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(range(1, len(binary_clf.loss_history) + 1), binary_clf.loss_history, color="#1f77b4", lw=2)
    ax.set_title("Binary Logistic Regression: Training Loss vs. Iteration", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration Number", fontsize=11)
    ax.set_ylabel(r"Cross-Entropy Loss $J(\theta)$ ($\lambda=0.1$)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig1_path = "fig1_binary_loss.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved Figure 1 to: {fig1_path}")

    # part 2.3: train and evaluate both models
    print("\n--- Part 2.3: Comparison of Two-Stage Classifier and Softmax Regression ---")
    # two-stage model
    two_stage_clf = TwoStageClassifier(eta=0.5, lam=0.01, max_iter=2000)
    two_stage_clf.fit(Xtr, ytr)

    ts_tr_pred = two_stage_clf.predict(Xtr)
    ts_te_pred = two_stage_clf.predict(Xte)
    ts_tr_acc = accuracy(ytr, ts_tr_pred)
    ts_te_acc = accuracy(yte, ts_te_pred)
    ts_cm_test = confusion_matrix(yte, ts_te_pred, K=3)

    # softmax model
    softmax_clf = SoftmaxRegression(eta=0.5, lam=0.01, max_iter=3000)
    softmax_clf.fit(Xtr, ytr, K=3)

    sm_tr_pred = softmax_clf.predict(Xtr)
    sm_te_pred = softmax_clf.predict(Xte)
    sm_tr_acc = accuracy(ytr, sm_tr_pred)
    sm_te_acc = accuracy(yte, sm_te_pred)
    sm_cm_test = confusion_matrix(yte, sm_te_pred, K=3)

    print(f"Two-Stage Classifier -> Train Acc: {ts_tr_acc:.4f}, Test Acc: {ts_te_acc:.4f}")
    print(f"Softmax Regression   -> Train Acc: {sm_tr_acc:.4f}, Test Acc: {sm_te_acc:.4f}\n")

    print_evaluation("Two-Stage Classifier (Test Set)", yte, ts_te_pred, ts_cm_test)
    print_evaluation("Softmax Regression (Test Set)", yte, sm_te_pred, sm_cm_test)

    # figure 2: scatter plot of petal length vs petal width
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    species_colors = {0: "#2ca02c", 1: "#1f77b4", 2: "#d62728"}
    species_markers = {0: "o", 1: "s", 2: "^"}

    for k, cname in enumerate(CLASS_NAMES):
        mask = (ytr == k)
        ax.scatter(
            Xtr_raw[mask, 2],
            Xtr_raw[mask, 3],
            c=species_colors[k],
            marker=species_markers[k],
            label=f"{cname.capitalize()} (n={mask.sum()})",
            s=45,
            edgecolor="k",
            alpha=0.85,
        )

    ax.set_title("Training Set: Petal Length vs. Petal Width by Species", fontsize=12, fontweight="bold")
    ax.set_xlabel("Petal Length (cm)", fontsize=11)
    ax.set_ylabel("Petal Width (cm)", fontsize=11)
    ax.legend(title="Species", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = "fig2_petal_scatter.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved Figure 2 to: {fig2_path}")

    # save fig 3: softmax loss curve
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(range(1, len(softmax_clf.loss_history) + 1), softmax_clf.loss_history, color="#d62728", lw=2)
    ax.set_title("Softmax Regression: Training Loss vs. Iteration", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration Number", fontsize=11)
    ax.set_ylabel(r"Multinomial Loss $J(W)$ ($\lambda=0.01$)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig3_path = "fig3_softmax_loss.png"
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved Figure 3 to: {fig3_path}")

    print("=" * 70)
    print("Execution completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
