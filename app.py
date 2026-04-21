import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Skill Gap Dashboard", layout="wide")
st.title("📊 Skill Gap Intelligence Dashboard")

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    supply = pd.read_csv("supply_clean.csv")
    demand = pd.read_csv("demand_clean.csv")
    return supply, demand

supply, demand = load_data()

# ── PREPROCESS ───────────────────────────────────────────────────────────────
def exp_to_num(x):
    if x == "0-1 year" or x == "0 - 1 year":
        return 1
    elif x == "2-5 years" or x == "2 - 5 years":
        return 3
    else:
        return 6

demand["exp_num"] = demand["Job Experience Required"].apply(exp_to_num)

# ── CLUSTERING ───────────────────────────────────────────────────────────────
@st.cache_data
def run_clustering(demand_df):
    X_cluster = demand_df[["Python", "SQL", "R", "exp_num"]]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    demand_df = demand_df.copy()
    demand_df["cluster"] = kmeans.fit_predict(X_cluster)
    cluster_summary = demand_df.groupby("cluster")[["Python", "SQL", "R"]].mean()
    cluster_names = cluster_summary.idxmax(axis=1).map(lambda x: f"{x}-Focused Jobs")
    demand_df["cluster_name"] = demand_df["cluster"].map(cluster_names)
    return demand_df, cluster_summary, cluster_names

demand, cluster_summary, cluster_names = run_clustering(demand)

# ── SIDEBAR NAVIGATION ───────────────────────────────────────────────────────
section = st.sidebar.radio("Navigate", [
    "Overview", "Skill Gap", "Experience Analysis",
    "Clustering", "Logistic Regression", "Explore"
])

# ── OVERVIEW ─────────────────────────────────────────────────────────────────
if section == "Overview":
    st.header("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Python — Supply / Demand",
              f"{supply['Python'].mean()*100:.1f}% / {demand['Python'].mean()*100:.1f}%")
    c2.metric("SQL — Supply / Demand",
              f"{supply['SQL'].mean()*100:.1f}% / {demand['SQL'].mean()*100:.1f}%")
    c3.metric("R — Supply / Demand",
              f"{supply['R'].mean()*100:.1f}% / {demand['R'].mean()*100:.1f}%")

    st.subheader("Raw Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Supply (Job Seekers)**")
        st.dataframe(supply.head())
    with col2:
        st.write("**Demand (Job Listings)**")
        st.dataframe(demand.head())

# ── SKILL GAP ────────────────────────────────────────────────────────────────
elif section == "Skill Gap":
    st.header("Skill Gap Analysis")

    supply_pct = supply[["Python", "SQL", "R"]].mean() * 100
    demand_pct = demand[["Python", "SQL", "R"]].mean() * 100

    comparison = pd.DataFrame({
        "Supply (%)": supply_pct,
        "Demand (%)": demand_pct,
        "Gap (%)": demand_pct - supply_pct
    })

    st.dataframe(comparison.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(7, 4))
    comparison[["Supply (%)", "Demand (%)"]].plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title("Skill Supply vs Demand")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(comparison.index, rotation=0)
    st.pyplot(fig)

    st.subheader("Skills by Job Role")
    role_skill = demand.groupby("Job Title")[["Python", "SQL", "R"]].mean()
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    role_skill.plot(kind="bar", ax=ax2)
    ax2.set_title("Skill Requirement by Job Role")
    ax2.set_ylabel("Proportion")
    ax2.set_xticklabels(role_skill.index, rotation=30, ha="right")
    st.pyplot(fig2)

# ── EXPERIENCE ANALYSIS ───────────────────────────────────────────────────────
elif section == "Experience Analysis":
    st.header("Experience Gap Analysis")

    exp_dist = demand["Job Experience Required"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    exp_dist.plot(kind="bar", ax=ax, color="mediumseagreen")
    ax.set_title("Job Experience Requirement Distribution")
    ax.set_ylabel("Count")
    ax.set_xticklabels(exp_dist.index, rotation=20, ha="right")
    st.pyplot(fig)

    st.subheader("Skill Demand by Experience Level")
    skill_exp = demand.groupby("Job Experience Required")[["Python", "SQL", "R"]].mean()
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    skill_exp.plot(kind="bar", ax=ax2)
    ax2.set_title("Skill Demand by Experience Level")
    ax2.set_ylabel("Proportion")
    ax2.set_xticklabels(skill_exp.index, rotation=20, ha="right")
    st.pyplot(fig2)

    st.subheader("Supply-Side: Skills by Age Group")
    age_skill = supply.groupby("Age")[["Python", "SQL", "R"]].mean()
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    age_skill.plot(kind="bar", ax=ax3)
    ax3.set_title("Skills by Age Group")
    ax3.set_ylabel("Proportion")
    ax3.set_xticklabels(age_skill.index, rotation=0)
    st.pyplot(fig3)

    st.subheader("Supply-Side: Skills by Education Level")
    edu_skill = supply.groupby("Education")[["Python", "SQL", "R"]].mean()
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    edu_skill.plot(kind="bar", ax=ax4)
    ax4.set_title("Skills by Education Level")
    ax4.set_ylabel("Proportion")
    ax4.set_xticklabels(edu_skill.index, rotation=0)
    st.pyplot(fig4)

# ── CLUSTERING ───────────────────────────────────────────────────────────────
elif section == "Clustering":
    st.header("K-Means Clustering (Demand Side)")

    st.subheader("Cluster Summary")
    summary = demand.groupby("cluster_name")[["Python", "SQL", "R"]].mean()
    st.dataframe(summary.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(7, 4))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Cluster Skill Profiles")
    ax.set_ylabel("Proportion")
    ax.set_xticklabels(summary.index, rotation=15, ha="right")
    st.pyplot(fig)

    st.subheader("PCA Cluster Projection")
    X_pca_input = demand[["Python", "SQL", "R"]]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_pca_input)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=demand["cluster"], cmap="Set1")
    ax2.set_title("Clusters (PCA Projection)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax2, label="Cluster")
    st.pyplot(fig2)

    st.write("**Cluster Counts:**")
    st.write(demand["cluster_name"].value_counts())

# ── LOGISTIC REGRESSION ───────────────────────────────────────────────────────
elif section == "Logistic Regression":
    st.header("Logistic Regression — Skill Prediction (Supply Side)")

    supply_encoded = pd.get_dummies(supply, columns=["Education", "Age"], drop_first=True)
    X = supply_encoded.drop(columns=["Python", "SQL", "R"])

    skill_choice = st.selectbox("Select Skill to Predict", ["Python", "SQL", "R"])
    y = supply_encoded[skill_choice]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc:.4f}")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

    st.subheader("Feature Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color="cornflowerblue")
    ax.set_title(f"Feature Coefficients for {skill_choice}")
    ax.set_xlabel("Coefficient Value")
    ax.invert_yaxis()
    st.pyplot(fig)

# ── EXPLORE ───────────────────────────────────────────────────────────────────
elif section == "Explore":
    st.header("Explore Jobs")

    st.subheader("Filter by Cluster")
    cluster_options = demand["cluster_name"].unique()
    selected = st.selectbox("Select Cluster", cluster_options)
    filtered = demand[demand["cluster_name"] == selected]
    st.write(f"**{len(filtered)} jobs found**")
    st.dataframe(filtered[["Job Title", "Job Experience Required", "Python", "SQL", "R"]])

    st.subheader("Filter by Skill")
    skill = st.selectbox("Required Skill", ["Python", "SQL", "R"])
    skill_filtered = demand[demand[skill] == 1]
    st.write(f"**{len(skill_filtered)} jobs require {skill}**")
    st.dataframe(skill_filtered[["Job Title", "Job Experience Required", "Python", "SQL", "R"]])
