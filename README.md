# 🎓 Student Performance Prediction

ML project that predicts a student's Final Performance Grade (A/B/C/D/F)
using study habits and academic features.

## 📁 Project Structure
```
student_performance/
├── data/
│   └── student_performance.csv   ← auto-generated dataset (600 records)
├── models/
│   ├── best_model.pkl            ← saved best ML model
│   ├── scaler.pkl                ← feature scaler
│   └── label_encoder.pkl         ← grade encoder
├── outputs/
│   ├── eda_report.png            ← EDA plots
│   ├── model_comparison.png      ← all 6 models compared
│   ├── confusion_matrix.png      ← best model results
│   └── feature_importance.png    ← which features matter most
├── generate_dataset.py           ← Step 1: create dataset
├── eda.py                        ← Step 2: explore data
├── train_models.py               ← Step 3: train & compare models
├── predict.py                    ← Step 4: make predictions
└── requirements.txt
```

## 🚀 How to Run (in order)

```bash
pip install -r requirements.txt

python generate_dataset.py   # create dataset
python eda.py                # explore data
python train_models.py       # train all models, pick best
python predict.py            # predict new student grades
```

## 🤖 Models Used
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- SVM (RBF kernel)
- Gradient Boosting

## 🎯 Features
| Feature | Description |
|---|---|
| Study_Hours_Per_Day | Daily study hours (0.5–10) |
| Attendance_Percentage | Class attendance % |
| Assignments_Completed | Assignments done (0–20) |
| Previous_Semester_Marks | Last semester score |
| Class_Participation | 1–5 scale |

**Target:** Final_Performance_Grade (A / B / C / D / F)
