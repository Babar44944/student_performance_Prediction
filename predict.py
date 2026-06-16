"""
predict.py
----------
Use the saved best model to predict a new student's grade
"""

import joblib
import numpy as np

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

def predict_grade(study_hours, attendance, assignments, prev_marks, class_part):
    """
    Parameters:
        study_hours  : float  (e.g. 5.0)
        attendance   : float  (e.g. 85.0)
        assignments  : int    (e.g. 15)
        prev_marks   : float  (e.g. 70.0)
        class_part   : int    (1-5)
    """
    X = np.array([[study_hours, attendance, assignments, prev_marks, class_part]])
    # Note: use scaler only if best model was a scaled model
    # For tree-based models scaler won't matter but we apply anyway
    try:
        X_sc = scaler.transform(X)
        pred = model.predict(X_sc)
    except:
        pred = model.predict(X)
    grade = le.inverse_transform(pred)[0]
    return grade

# ── Demo ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("   STUDENT GRADE PREDICTOR")
    print("=" * 45)

    test_students = [
        {"study_hours": 7,   "attendance": 92, "assignments": 18, "prev_marks": 80, "class_part": 5},
        {"study_hours": 2,   "attendance": 45, "assignments": 5,  "prev_marks": 40, "class_part": 1},
        {"study_hours": 4.5, "attendance": 70, "assignments": 12, "prev_marks": 60, "class_part": 3},
    ]

    for i, s in enumerate(test_students, 1):
        grade = predict_grade(**s)
        print(f"\nStudent {i}: {s}")
        print(f"  ➜  Predicted Grade: {grade}")

    print("\n✅ Prediction demo complete!")
    print("\nTo predict your own student:")
    print("  grade = predict_grade(study_hours=5, attendance=80,")
    print("                        assignments=14, prev_marks=65, class_part=4)")
