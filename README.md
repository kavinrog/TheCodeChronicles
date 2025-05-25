# 📊 SHAP Explanation for XGBoost on California Housing Data

This README explains the SHAP (SHapley Additive exPlanations) visualizations for an **XGBoost model** predicting **house prices** based on the **California Housing dataset**.

<<<<<<< HEAD
## 🔥 Contributions
1. **Simple Neural Network Without Machine Learning Libraries**
  - Implemented a basic neural network from scratch using only Python's built-in modules.
2. **SHAP Explanation for XGBoost on California Housing Data**
  - SHAP visualizations for XGBoost on the California Housing dataset reveal how features like MedInc, Latitude, and Longitude influence house price predictions.
3. **Gap Statistic Method for Finding Optimal Clusters**
  - A statistical approach to determine the optimal number of clusters by comparing actual clustering performance with a random baseline.
4. **Linear & Logistic Regression from Scratch**
  - Linear Regression predicts continuous values using a straight-line equation optimized by gradient descent, while Logistic Regression performs binary classification by applying the sigmoid function to a linear model.
5. **Support Vector Machine (SVM) from Scratch**
  - Implements a Support Vector Machine (SVM) classifier using Stochastic Gradient Descent (SGD)for binary classification. The model is trained using the hinge loss function and L2 regularization.
6. **Regression Analysis with Scikit-Learn**
  - Implements various regression techniques using Scikit-Learn and other relevant Python libraries.
7. **Deep Neural Network from Scratchn**
  - This project implements two versions of a Deep Neural Network (DNN) from scratch using NumPy
=======
## 🏡 Dataset Overview
The dataset includes features like:
- **MedInc** (Median Income)
- **HouseAge** (House Age)
- **AveRooms** (Average Rooms per House)
- **Latitude & Longitude** (Location)
- **Population & Occupancy**
>>>>>>> 8533f4c (Add readme)

The model predicts house prices using these features.

---

## 🔹 **SHAP Decision Plot**
![SHAP Decision Plot](./Decision%20Plot.png)

### 📌 What it Shows:
- How **each feature affects a single prediction**.
- The **blue line** traces how features push the price up or down.
- **Key Influence**: **MedInc, Latitude, and Longitude** have the biggest impact.

---

## 🔹 **SHAP Force Plot**
![SHAP Force Plot](./Force%20Plot.png)

### 📌 What it Shows:
- **How a prediction was made for one house**.
- **Blue** pushes the price **down**, **red** pushes it **up**.
- **MedInc, Latitude, and AveOccup** strongly influence this prediction.

---

## 🔹 **SHAP Summary Plot**
![SHAP Summary Plot](./Summary%20Plot.png)

### 📌 What it Shows:
- **Overall feature importance across all predictions**.
- **Red = High feature value, Blue = Low feature value**.
- **MedInc, Latitude, and Longitude** are the most important.

---

## 🎯 **Conclusion**
- **MedInc (Income) is the strongest predictor** of house prices.
- **Location (Latitude & Longitude) also plays a big role**.
- SHAP makes model decisions **explainable and transparent**.

🚀 **SHAP helps us understand how AI makes predictions!**
