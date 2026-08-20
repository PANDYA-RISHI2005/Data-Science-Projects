# ==========================================
# STUDENT PERFORMANCE ANALYZER
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

df = pd.read_csv(r"C:\Users\rishi\Downloads\student_performance_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# ------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------
# 3. CHECK DUPLICATES
# ------------------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# ------------------------------------------
# 4. BASIC STATISTICS
# ------------------------------------------

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------------------
# 5. AVERAGE SCORES
# ------------------------------------------

print("\nAverage Math Score:",
      df["Math_Score"].mean())

print("Average Science Score:",
      df["Science_Score"].mean())

print("Average English Score:",
      df["English_Score"].mean())

print("Average Final Score:",
      df["Final_Score"].mean())

print("Average Attendance:",
      df["Attendance"].mean())

# ------------------------------------------
# 6. TOP 10 STUDENTS
# ------------------------------------------

top_students = df.sort_values(
    by="Final_Score",
    ascending=False
).head(10)

print("\nTop 10 Students:")
print(top_students[
    ["Student_ID", "Final_Score", "Performance"]
])

# ------------------------------------------
# 7. PERFORMANCE DISTRIBUTION
# ------------------------------------------

print("\nPerformance Distribution:")
print(df["Performance"].value_counts())

# ------------------------------------------
# 8. SUBJECT-WISE AVERAGE
# ------------------------------------------

subjects = [
    "Math_Score",
    "Science_Score",
    "English_Score"
]

average_scores = df[subjects].mean()

print("\nSubject-wise Average:")
print(average_scores)

# ------------------------------------------
# 9. BAR CHART
# ------------------------------------------

plt.figure(figsize=(8, 5))

average_scores.plot(kind="bar")

plt.title("Average Score by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# ------------------------------------------
# 10. PERFORMANCE COUNT
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Performance"
)

plt.title("Student Performance Distribution")
plt.xlabel("Performance")
plt.ylabel("Number of Students")

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

# ------------------------------------------
# 11. STUDY HOURS VS FINAL SCORE
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 12. ATTENDANCE VS FINAL SCORE
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Attendance",
    y="Final_Score"
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 13. CORRELATION MATRIX
# ------------------------------------------

numeric_columns = [
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Math_Score",
    "Science_Score",
    "English_Score",
    "Final_Score"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

# ------------------------------------------
# 14. CORRELATION HEATMAP
# ------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Student Performance Correlation")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 15. FIND WEAK STUDENTS
# ------------------------------------------

weak_students = df[
    df["Final_Score"] < 40
]

print("\nStudents Needing Improvement:")

print(
    weak_students[
        [
            "Student_ID",
            "Study_Hours",
            "Attendance",
            "Final_Score",
            "Performance"
        ]
    ]
)

# ------------------------------------------
# 16. GENDER-WISE PERFORMANCE
# ------------------------------------------

gender_performance = df.groupby(
    "Gender"
)["Final_Score"].mean()

print("\nGender-wise Average Performance:")
print(gender_performance)

# ------------------------------------------
# 17. STUDY HOURS GROUP ANALYSIS
# ------------------------------------------

study_analysis = df.groupby(
    "Study_Hours"
)["Final_Score"].mean()

print("\nAverage Score by Study Hours:")
print(study_analysis)

# ------------------------------------------
# 18. SAVE ANALYZED DATA
# ------------------------------------------

df.to_csv(
    "student_performance_analyzed.csv",
    index=False
)

print("\nAnalysis completed successfully!")
print("Analyzed dataset saved.")