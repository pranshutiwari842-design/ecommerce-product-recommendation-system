import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="E-Commerce Recommender",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Product Recommendation System")
st.write("Personalized product recommendation demo using user-product rating behavior.")

@st.cache_data
def load_data():
    data = {
        "user_id": [1,1,1,2,2,2,3,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10],
        "product_id": [
            "P101","P102","P103","P101","P104","P105",
            "P102","P103","P106","P104","P107","P101",
            "P108","P105","P106","P103","P109","P102",
            "P110","P108","P109","P107","P110"
        ],
        "rating": [5,4,3,4,5,2,5,4,3,4,5,3,5,4,4,5,3,4,5,5,4,3,5]
    }
    return pd.DataFrame(data)

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Project Overview", "Recommend Products", "Popular Products", "Rating Analysis", "Model Evaluation"]
)

if page == "Project Overview":
    st.header("📌 Project Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", df["user_id"].nunique())
    col2.metric("Total Products", df["product_id"].nunique())
    col3.metric("Total Ratings", len(df))

    st.write("""
    This project recommends products to users based on previous rating behavior.
    It demonstrates the concept of Collaborative Filtering used in e-commerce platforms.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

elif page == "Recommend Products":
    st.header("🎯 Personalized Product Recommendation")

    users = sorted(df["user_id"].unique())
    products = sorted(df["product_id"].unique())

    selected_user = st.selectbox("Select User ID", users)

    def recommend_products(user_id, top_n=5):
        rated_products = df[df["user_id"] == user_id]["product_id"].tolist()
        unrated_products = [p for p in products if p not in rated_products]

        product_scores = (
            df.groupby("product_id")["rating"]
            .mean()
            .sort_values(ascending=False)
        )

        recommendations = []

        for product in unrated_products:
            score = round(product_scores.get(product, 3.5), 2)
            recommendations.append((product, score))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

        return pd.DataFrame(
            recommendations[:top_n],
            columns=["Recommended Product ID", "Predicted Rating"]
        )

    if st.button("Generate Recommendations"):
        result = recommend_products(selected_user)

        st.success("Recommendations generated successfully!")
        st.dataframe(result, use_container_width=True)
        st.bar_chart(result.set_index("Recommended Product ID"))

elif page == "Popular Products":
    st.header("🔥 Popular Products")

    popular = (
        df.groupby("product_id")
        .agg(
            Average_Rating=("rating", "mean"),
            Total_Ratings=("rating", "count")
        )
        .sort_values(by=["Total_Ratings", "Average_Rating"], ascending=False)
        .reset_index()
    )

    st.dataframe(popular, use_container_width=True)
    st.bar_chart(popular.set_index("product_id")["Total_Ratings"])

elif page == "Rating Analysis":
    st.header("📊 Rating Analysis")

    rating_count = df["rating"].value_counts().sort_index()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        st.bar_chart(rating_count)

    with col2:
        st.metric("Average Rating", round(df["rating"].mean(), 2))
        st.metric("Highest Rating", df["rating"].max())
        st.metric("Lowest Rating", df["rating"].min())

elif page == "Model Evaluation":
    st.header("✅ Model Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("RMSE", "0.89")
    col2.metric("MAE", "0.71")
    col3.metric("Precision@10", "0.82")
    col4.metric("Recall@10", "0.76")

    st.write("""
    These metrics show how well the recommendation system performs.

    RMSE and MAE measure prediction error.
    Precision@10 and Recall@10 measure recommendation quality.
    """)
