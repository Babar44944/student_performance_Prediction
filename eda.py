"""
eda.py
------
Exploratory Data Analysis - Student Performance Dataset
Saves all plots to outputs/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/student_performance.csv")

print("=" * 55)
print("        STUDENT PERFORMANCE - EDA REPORT")
print("=" * 55)
print(f"\n📊 Dataset Shape : {df.shape}")
print(f"🔢 Features      : {list(df.columns)}")
print(f"\n📋 First 5 rows:\n{df.head()}")
print(f"\n📈 Statistics:\n{df.describe()}")
print(f"\n🎯 Grade Distribution:\n{df['Final_Performance_Grade'].value_counts()}")
print(f"\n❓ Missing Values:\n{df.isnull().sum()}")

# ── Plot 1: Grade Distribution ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Student Performance - EDA", fontsize=18, fontweight="bold")

grade_order = ["A", "B", "C", "D", "F"]
colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
counts = df["Final_Performance_Grade"].value_counts().reindex(grade_order)
axes[0, 0].bar(grade_order, counts, color=colors, edgecolor="white", linewidth=1.5)
axes[0, 0].set_title("Grade Distribution", fontweight="bold")
axes[0, 0].set_xlabel("Grade")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(counts):
    axes[0, 0].text(i, v + 2, str(v), ha="center", fontweight="bold")

# ── Plot 2: Study Hours vs Final Score ───────────────────────
scatter_colors = {"A":"#2ecc71","B":"#3498db","C":"#f39c12","D":"#e67e22","F":"#e74c3c"}
for grade in grade_order:
    sub = df[df["Final_Performance_Grade"] == grade]
    axes[0, 1].scatter(sub["Study_Hours_Per_Day"], sub["Final_Score"],
                       label=grade, color=scatter_colors[grade], alpha=0.6, s=30)
axes[0, 1].set_title("Study Hours vs Final Score", fontweight="bold")
axes[0, 1].set_xlabel("Study Hours Per Day")
axes[0, 1].set_ylabel("Final Score")
axes[0, 1].legend(title="Grade")

# ── Plot 3: Attendance vs Final Score ────────────────────────
axes[0, 2].scatter(df["Attendance_Percentage"], df["Final_Score"],
                   alpha=0.5, color="#9b59b6", s=30)
axes[0, 2].set_title("Attendance vs Final Score", fontweight="bold")
axes[0, 2].set_xlabel("Attendance %")
axes[0, 2].set_ylabel("Final Score")

# ── Plot 4: Correlation Heatmap ──────────────────────────────
numeric_cols = ["Study_Hours_Per_Day","Attendance_Percentage",
                "Assignments_Completed","Previous_Semester_Marks",
                "Class_Participation","Final_Score"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[1, 0], linewidths=0.5)
axes[1, 0].set_title("Correlation Heatmap", fontweight="bold")

# ── Plot 5: Boxplot per Grade ────────────────────────────────
grade_data = [df[df["Final_Performance_Grade"] == g]["Final_Score"].values
              for g in grade_order]
bp = axes[1, 1].boxplot(grade_data, labels=grade_order, patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_title("Score Distribution by Grade", fontweight="bold")
axes[1, 1].set_xlabel("Grade")
axes[1, 1].set_ylabel("Final Score")

# ── Plot 6: Previous Marks vs Final Score ────────────────────
axes[1, 2].scatter(df["Previous_Semester_Marks"], df["Final_Score"],
                   alpha=0.5, color="#1abc9c", s=30)
axes[1, 2].set_title("Previous Marks vs Final Score", fontweight="bold")
axes[1, 2].set_xlabel("Previous Semester Marks")
axes[1, 2].set_ylabel("Final Score")

plt.tight_layout()
plt.savefig("outputs/eda_report.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ EDA plots saved → outputs/eda_report.png")
