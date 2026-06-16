"""
generate_dataset.py
-------------------
Synthetic Student Performance Dataset Generator
500+ records with realistic correlations
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 600

study_hours = np.round(np.clip(np.random.normal(4, 1.5, N), 0.5, 10), 1)
attendance  = np.round(np.clip(np.random.normal(75, 15, N), 30, 100), 1)
assignments = np.random.randint(0, 21, N)
prev_marks  = np.round(np.clip(np.random.normal(60, 15, N), 20, 100), 1)
class_part  = np.random.randint(1, 6, N)

score = (
    study_hours * 3.5 +
    attendance  * 0.3 +
    assignments * 1.2 +
    prev_marks  * 0.4 +
    class_part  * 2.0 +
    np.random.normal(0, 5, N)
)
score = (score - score.min()) / (score.max() - score.min()) * 100

def assign_grade(s):
    if s >= 80: return "A"
    elif s >= 65: return "B"
    elif s >= 50: return "C"
    elif s >= 35: return "D"
    else: return "F"

grades = [assign_grade(s) for s in score]

df = pd.DataFrame({
    "Study_Hours_Per_Day"    : study_hours,
    "Attendance_Percentage"  : attendance,
    "Assignments_Completed"  : assignments,
    "Previous_Semester_Marks": prev_marks,
    "Class_Participation"    : class_part,
    "Final_Score"            : np.round(score, 2),
    "Final_Performance_Grade": grades
})

df.to_csv("data/student_performance.csv", index=False)
print(f"✅ Dataset saved! Shape: {df.shape}")
print(df["Final_Performance_Grade"].value_counts())
print(df.head())
