import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="E-Commerce Recommender",
    page_icon="🛒",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f4e79;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.card {
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric-card {
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    padding: 18px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Files ----------
@st.cache_data
def load_data():
    return pd.read_csv("models/processed_data.csv")

@st.cache_resource
def load_model():
    with open("models/recommender_model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_metrics():
    with open("models/metrics.pkl", "rb") as file:
        return pickle.load(file)

df = load_data()
model = load_model()
metrics = load_metrics()

# ---------- Sidebar ----------
st.sidebar.title("🛒 Recommendation System")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Project Overview",
        "🎯 Recommend Products",
        "🔥 Popular Products",
        "📊 Rating Analysis",
        "✅ Model Evaluation",
        "🧠 How It Works"
    ]
)

st.sidebar.info("""
This project suggests products based on user rating behavior using SVD Matrix Factorization.
""")

# ---------- Page 1 ----------
if page == "🏠 Project Overview":
    st.markdown('<div class="big-title">E-Commerce Product Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A machine learning project that recommends products based on user behavior.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", df["user_id"].nunique())
    col2.metric("Total Products", df["product_id"].nunique())
    col3.metric("Total Ratings", len(df))

    st.markdown("### 📌 Project Objective")
    st.write("""
    The goal of this project is to suggest products to users based on their previous ratings.
    This helps e-commerce platforms improve personalization and user engagement.
    """)

    st.markdown("### 🧰 Tech Stack")
    st.write("Python | Pandas | NumPy | Scikit-Surprise | Streamlit | Machine Learning")

    st.markdown("### 📂 Sample Dataset")
    st.dataframe(df.head(10), use_container_width=True)

# ---------- Page 2 ----------
elif page == "🎯 Recommend Products":
    st.title("🎯 Personalized Product Recommendation")

    users = sorted(df["user_id"].unique())
    products = sorted(df["product_id"].unique())

    selected_user = st.selectbox("Select User ID", users)

    st.info("After selecting a user, the system predicts ratings for products the user has not rated yet.")

    def recommend_products(user_id, top_n=10):
        rated_products = df[df["user_id"] == user_id]["product_id"].tolist()
        unrated_products = [p for p in products if p not in rated_products]

        predictions = []

        for product_id in unrated_products[:3000]:
            pred = model.predict(user_id, product_id)
            predictions.append((product_id, round(pred.est, 2)))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(
            predictions[:top_n],
            columns=["Recommended Product ID", "Predicted Rating"]
        )

    if st.button("Generate Recommendations"):
        result_df = recommend_products(selected_user)

        st.success("Top 10 products recommended successfully!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(result_df, use_container_width=True)

        with col2:
            st.metric("Selected User", selected_user)
            st.metric("Recommendations", "10")

        st.markdown("### 📊 Recommendation Score Chart")
        st.bar_chart(result_df.set_index("Recommended Product ID"))

# ---------- Page 3 ----------
elif page == "🔥 Popular Products":
    st.title("🔥 Popular Products")

    popular = (
        df.groupby("product_id")
        .agg(
            Average_Rating=("rating", "mean"),
            Total_Ratings=("rating", "count")
        )
        .sort_values(by=["Total_Ratings", "Average_Rating"], ascending=False)
        .head(10)
        .reset_index()
    )

    st.write("These products are popular because they received the highest number of ratings.")

    st.dataframe(popular, use_container_width=True)

    st.markdown("### 📊 Most Rated Products")
    st.bar_chart(popular.set_index("product_id")["Total_Ratings"])

# ---------- Page 4 ----------
elif page == "📊 Rating Analysis":
    st.title("📊 Rating Analysis")

    col1, col2 = st.columns(2)

    rating_count = df["rating"].value_counts().sort_index()

    with col1:
        st.markdown("### Rating Distribution")
        st.bar_chart(rating_count)

    with col2:
        st.markdown("### Average Rating")
        st.metric("Average Rating", round(df["rating"].mean(), 2))
        st.metric("Highest Rating", df["rating"].max())
        st.metric("Lowest Rating", df["rating"].min())

    st.markdown("### Explanation")
    st.write("""
    This section helps us understand user behavior.
    If most ratings are high, it means users are generally satisfied with products.
    """)

# ---------- Page 5 ----------
elif page == "✅ Model Evaluation":
    st.title("✅ Model Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("RMSE", round(metrics["RMSE"], 3))
    col2.metric("MAE", round(metrics["MAE"], 3))
    col3.metric("Precision@10", round(metrics["Precision@10"], 3))
    col4.metric("Recall@10", round(metrics["Recall@10"], 3))

    st.markdown("### Easy Explanation")

    st.write("""
    **RMSE** tells how much error exists between actual and predicted ratings.

    **MAE** tells average prediction error.

    **Precision@10** tells how many recommended products are actually useful.

    **Recall@10** tells how many useful products were successfully recommended.
    """)

# ---------- Page 6 ----------
elif page == "🧠 How It Works":
    st.title("🧠 How Recommendation System Works")

    st.markdown("""
    ### Step 1: User gives product ratings
    Example: User A rated Product 1 as 5 stars.

    ### Step 2: System finds hidden patterns
    It checks which users have similar rating behavior.

    ### Step 3: Matrix Factorization
    SVD breaks the user-product rating matrix into smaller hidden feature matrices.

    ### Step 4: Predict missing ratings
    The model predicts how much a user may like an unseen product.

    ### Step 5: Recommend top products
    Products with highest predicted ratings are shown to the user.
    """)

    st.success("This is the same logic used by platforms like Amazon, Netflix, and Flipkart.")