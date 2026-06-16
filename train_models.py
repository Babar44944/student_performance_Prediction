"""
train_models.py
---------------
Full ML Pipeline - Multiple Models + Best Model Selection
Models: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ── Load Data ────────────────────────────────────────────────
df = pd.read_csv("data/student_performance.csv")
FEATURES = ["Study_Hours_Per_Day", "Attendance_Percentage",
            "Assignments_Completed", "Previous_Semester_Marks", "Class_Participation"]
TARGET = "Final_Performance_Grade"

X = df[FEATURES]
le = LabelEncoder()
y = le.fit_transform(df[TARGET])   # A=0, B=1, C=2, D=3, F=4

# ── Train/Test Split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Define Models ────────────────────────────────────────────
models = {
    "Logistic Regression"   : LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors"   : KNeighborsClassifier(n_neighbors=7),
    "Decision Tree"         : DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest"         : RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM"                   : SVC(kernel="rbf", probability=True, random_state=42),
    "Gradient Boosting"     : GradientBoostingClassifier(n_estimators=200, random_state=42),
}

# Scaled models
scaled_models = {"Logistic Regression", "K-Nearest Neighbors", "SVM"}

print("=" * 65)
print("          STUDENT PERFORMANCE - MODEL TRAINING")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    Xtr = X_train_sc if name in scaled_models else X_train.values
    Xte = X_test_sc  if name in scaled_models else X_test.values

    # Cross-validation
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    test_acc = accuracy_score(y_test, y_pred)

    results[name] = {
        "model"      : model,
        "cv_mean"    : cv_scores.mean(),
        "cv_std"     : cv_scores.std(),
        "test_acc"   : test_acc,
        "y_pred"     : y_pred,
        "scaled"     : name in scaled_models,
    }
    print(f"\n🔹 {name}")
    print(f"   CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")

# ── Best Model ───────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["test_acc"])
best      = results[best_name]
print(f"\n{'='*65}")
print(f"  🏆 BEST MODEL : {best_name}")
print(f"     Test Accuracy : {best['test_acc']:.4f}")
print(f"{'='*65}")
print(f"\n📋 Classification Report:\n")
print(classification_report(y_test, best["y_pred"], target_names=le.classes_))

# ── Save Best Model ──────────────────────────────────────────
joblib.dump(best["model"], "models/best_model.pkl")
joblib.dump(scaler,         "models/scaler.pkl")
joblib.dump(le,             "models/label_encoder.pkl")
print("✅ Best model saved → models/best_model.pkl")

# ── Plot 1: Model Comparison ─────────────────────────────────
names     = list(results.keys())
test_accs = [results[n]["test_acc"] for n in names]
cv_means  = [results[n]["cv_mean"]  for n in names]
cv_stds   = [results[n]["cv_std"]   for n in names]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Model Comparison - Student Performance", fontsize=16, fontweight="bold")

bar_colors = ["#e74c3c" if n == best_name else "#3498db" for n in names]
bars = axes[0].barh(names, test_accs, color=bar_colors, edgecolor="white")
axes[0].set_xlabel("Test Accuracy")
axes[0].set_title("Test Accuracy per Model (🔴 = Best)", fontweight="bold")
axes[0].set_xlim(0, 1.05)
for bar, acc in zip(bars, test_accs):
    axes[0].text(acc + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{acc:.4f}", va="center", fontweight="bold")

axes[1].barh(names, cv_means, xerr=cv_stds, color=bar_colors,
             edgecolor="white", capsize=4)
axes[1].set_xlabel("CV Accuracy (mean ± std)")
axes[1].set_title("5-Fold CV Accuracy per Model", fontweight="bold")
axes[1].set_xlim(0, 1.05)

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 2: Confusion Matrix (Best Model) ────────────────────
cm = confusion_matrix(y_test, best["y_pred"])
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3: Feature Importance (if RF or GB) ─────────────────
if hasattr(best["model"], "feature_importances_"):
    fi = best["model"].feature_importances_
    sorted_idx = np.argsort(fi)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(FEATURES)))
    ax.barh([FEATURES[i] for i in sorted_idx],
            [fi[i] for i in sorted_idx],
            color=colors_fi, edgecolor="white")
    ax.set_title(f"Feature Importances — {best_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Feature importance plot saved → outputs/feature_importance.png")

print("✅ Model comparison  saved → outputs/model_comparison.png")
print("✅ Confusion matrix  saved → outputs/confusion_matrix.png")
print("\n🎉 Training Complete!")
