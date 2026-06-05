# 🛒 E-Commerce Product Recommendation System

## 📌 Project Overview

The E-Commerce Product Recommendation System is a Machine Learning project designed to recommend products to users based on their previous interactions and ratings. The system analyzes user behavior and predicts products that a user is likely to purchase or rate highly.

This project uses Collaborative Filtering and Matrix Factorization (SVD) techniques to generate personalized recommendations. A Streamlit web application is developed to provide an interactive dashboard for users.

---

## 🎯 Objective

The primary objective of this project is to:

* Suggest relevant products to users.
* Improve user experience through personalization.
* Increase customer engagement and sales.
* Demonstrate recommendation system concepts used in real-world e-commerce platforms such as Amazon, Flipkart, and Netflix.

---

## 📂 Dataset

Dataset Used: Amazon Fine Food Reviews Dataset

Source: Kaggle

### Dataset Columns Used

| Column    | Description               |
| --------- | ------------------------- |
| UserId    | Unique User Identifier    |
| ProductId | Unique Product Identifier |
| Score     | Product Rating (1-5)      |
| Summary   | Review Summary            |

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Scikit-Surprise
* Streamlit

### Machine Learning Techniques

* Collaborative Filtering
* Matrix Factorization
* Singular Value Decomposition (SVD)

---

## 🔄 Project Workflow

### 1. Data Collection

The Amazon Fine Food Reviews dataset is loaded into the system.

### 2. Data Preprocessing

* Removed missing values
* Selected relevant columns
* Renamed columns
* Filtered inactive users and products

### 3. Exploratory Data Analysis

* User activity analysis
* Product popularity analysis
* Rating distribution analysis

### 4. Model Building

The recommendation model is built using SVD Matrix Factorization.

The model learns hidden relationships between users and products and predicts ratings for unseen products.

### 5. Recommendation Generation

For a selected user:

* Previously rated products are excluded.
* Predicted ratings are calculated.
* Top products are recommended.

### 6. Evaluation

The model performance is measured using:

* RMSE
* MAE
* Precision@10
* Recall@10
* NDCG@10

### 7. Deployment

The recommendation system is deployed using Streamlit.

---

## 📊 Dashboard Features

### 🏠 Project Overview

* Total Users
* Total Products
* Total Ratings
* Dataset Preview

### 🎯 Product Recommendation

* User Selection
* Top Recommended Products
* Predicted Ratings
* Recommendation Visualization

### 🔥 Popular Products

* Top Rated Products
* Most Reviewed Products

### 📈 Rating Analysis

* Rating Distribution
* Average Rating
* User Behavior Analysis

### ✅ Model Evaluation

* RMSE
* MAE
* Precision@10
* Recall@10

### 🧠 Recommendation Logic

* Collaborative Filtering Explanation
* SVD Matrix Factorization Workflow

---

## 🤖 Machine Learning Model

### Collaborative Filtering

Collaborative Filtering recommends products based on similarities between user preferences.

### Matrix Factorization (SVD)

Singular Value Decomposition breaks the User-Product Matrix into latent factors and predicts missing ratings.

Benefits:

* Better personalization
* Handles sparse data
* Scalable for large datasets

---

## 📏 Evaluation Metrics

### RMSE

Measures prediction error.

### MAE

Measures average absolute error.

### Precision@10

Measures relevance of recommended products.

### Recall@10

Measures coverage of relevant products.

### NDCG@10

Measures ranking quality of recommendations.

---

## 📂 Project Structure

```text
ecommerce-recommendation-system/
│
├── data/
│   └── Reviews.csv
│
├── models/
│   ├── recommender_model.pkl
│   ├── processed_data.csv
│   └── metrics.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/ecommerce-recommendation-system.git
```

### Move to Project Directory

```bash
cd ecommerce-recommendation-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Run Streamlit App

```bash
streamlit run app.py
```
## 🚀 Live Demo

Streamlit App:
https://ecommerce-appuct-recommendation-system-jbybhzeyblannifvbci67b.streamlit.app/


---

## 📈 Business Benefits

* Personalized shopping experience
* Increased customer retention
* Better product discovery
* Improved conversion rates
* Enhanced customer satisfaction

---

## 🌍 Real-World Applications

* Amazon
* Flipkart
* Myntra
* Netflix
* Spotify
* YouTube

---

## 🔮 Future Enhancements

* Deep Learning Recommendation Models
* Neural Collaborative Filtering
* Hybrid Recommendation System
* Product Image-Based Recommendations
* Real-Time Recommendations
* Cloud Deployment

---

## 👨‍💻 Author

Pranshu Tiwari

BCA (Data Science)

Skills:

* Python
* SQL
* Power BI
* Machine Learning
* Data Analysis
* Recommendation Systems

LinkedIn:
https://www.linkedin.com/in/pranshu-tiwari-6113a7350
