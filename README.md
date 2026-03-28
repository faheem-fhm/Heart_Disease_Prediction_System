# ❤️ Heart Disease Prediction using Gaussian Naïve Bayes

## 📌 Project Description
This project uses Machine Learning to predict whether a patient has heart disease based on medical data.  
The model helps in **early detection**, which can improve treatment and save lives.

---

## 🎯 Objective
- Predict heart disease (0 = No Disease, 1 = Disease)
- Assist doctors in early diagnosis
- Build a simple and efficient ML model

---

## 📊 Dataset Information
- Total Records: 918 patients  
- Features: 12  
- Dataset File: `heart.csv`

### 🔑 Features Used:
- Age  
- Sex  
- ChestPainType  
- RestingBP  
- Cholesterol  
- FastingBS  
- RestingECG  
- MaxHR  
- ExerciseAngina  
- ST_Slope  

### 🆕 Feature Engineering:
- **RiskScore = Cholesterol + RestingBP**

---

## 🤖 Algorithm Used
- Gaussian Naïve Bayes  
- Fast and efficient  
- Works well with small/medium datasets  
- Provides probability-based predictions  

---

## ⚙️ Steps Involved
1. Load Dataset  
2. Data Cleaning (handle missing values)  
3. Label Encoding (convert categorical to numeric)  
4. Feature Engineering (RiskScore)  
5. Train-Test Split (80% / 20%)  
6. Model Training using GaussianNB  
7. Prediction and Evaluation  

---

## 📈 Model Performance
- Accuracy: **85.3%**  
- Precision: **85.1%**  
- Recall: **87.8%**  
- F1 Score: **86.4%**

---

## 📊 Confusion Matrix
|                | Predicted 0 | Predicted 1 |
|----------------|------------|------------|
| Actual 0       | 71         | 15         |
| Actual 1       | 12         | 86         |

---

## 🧪 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn
- Streamlit

---

## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install pandas numpy scikit-learn Seaborn Streamlit
