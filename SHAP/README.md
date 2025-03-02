# 📊 SHAP Explanation for XGBoost on California Housing Data

This README explains the SHAP (SHapley Additive exPlanations) visualizations for an **XGBoost model** predicting **house prices** based on the **California Housing dataset**.

## 🏡 Dataset Overview
The dataset includes features like:
- **MedInc** (Median Income)
- **HouseAge** (House Age)
- **AveRooms** (Average Rooms per House)
- **Latitude & Longitude** (Location)
- **Population & Occupancy**

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
