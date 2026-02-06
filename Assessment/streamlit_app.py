import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from io import StringIO

# Page configuration
st.set_page_config(page_title="Customer Clustering Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# Load data and models
@st.cache_resource
def load_data_and_models():
    # Load customer data with clusters
    df = pd.read_csv('customer_clusters.csv')
    
    # Load trained models
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('cluster_segments.pkl', 'rb') as f:
        cluster_segments = pickle.load(f)
    
    return df, kmeans, scaler, cluster_segments

df, kmeans, scaler, cluster_segments = load_data_and_models()

# Title
st.title("🎯 Customer Clustering Analysis Dashboard")
st.markdown("---")

# Sidebar navigation
page = st.sidebar.radio("Navigation", 
    ["📊 Overview", "📈 Cluster Analysis", "👥 Customer Details", "🔮 Predict Cluster"])

# Define color mapping
colors_map = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.header("Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    
    with col2:
        st.metric("Number of Clusters", len(df['Cluster'].unique()))
    
    with col3:
        st.metric("Avg Annual Income", f"₹{df['AnnualIncome'].mean():,.0f}")
    
    with col4:
        st.metric("Avg Total Spent", f"₹{df['TotalSpent'].mean():,.0f}")
    
    st.markdown("---")
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution by Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = [colors_map[int(cluster)] for cluster in cluster_counts.index]
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color=bar_colors, 
                     edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
        ax.set_title('Customer Count per Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels([f'Cluster {i}\n{cluster_segments[i]}' for i in cluster_counts.index], fontsize=9)
        
        for bar, cluster in zip(bars, cluster_counts.index):
            height = bar.get_height()
            pct = (height / len(df)) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Cluster Summary Statistics")
        summary_data = []
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster]
            summary_data.append({
                'Cluster': f"{cluster}",
                'Segment': cluster_segments[cluster],
                'Count': len(cluster_data),
                'Percentage': f"{(len(cluster_data)/len(df)*100):.1f}%",
                'Avg Spend': f"₹{cluster_data['TotalSpent'].mean():,.0f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 2: CLUSTER ANALYSIS
# ============================================================================
elif page == "📈 Cluster Analysis":
    st.header("Detailed Cluster Analysis")
    
    # Filter by cluster
    selected_clusters = st.multiselect(
        "Select Clusters to Display",
        options=sorted(df['Cluster'].unique()),
        default=sorted(df['Cluster'].unique()),
        format_func=lambda x: f"Cluster {x}: {cluster_segments[x]}"
    )
    
    df_filtered = df[df['Cluster'].isin(selected_clusters)]
    
    st.markdown("---")
    
    # Average Spending per Cluster
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Total Spending per Cluster")
        avg_spending = df_filtered.groupby('Cluster')['TotalSpent'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = [colors_map[int(cluster)] for cluster in avg_spending.index]
        bars = ax.bar(avg_spending.index, avg_spending.values, color=bar_colors, 
                     edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Total Spent (₹)', fontsize=12, fontweight='bold')
        ax.set_title('Average Spending per Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(avg_spending)))
        
        for bar, cluster in zip(bars, avg_spending.index):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'₹{height:,.0f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Average App Usage per Cluster")
        avg_app_usage = df_filtered.groupby('Cluster')['AppTimeMinutes'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = [colors_map[int(cluster)] for cluster in avg_app_usage.index]
        bars = ax.bar(avg_app_usage.index, avg_app_usage.values, color=bar_colors, 
                     edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average App Usage (Minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Average App Usage per Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(avg_app_usage)))
        
        for bar, cluster in zip(bars, avg_app_usage.index):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}m',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Income vs Spending scatter plot
    st.subheader("Income vs Spending - Cluster Visualization")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for cluster in sorted(df_filtered['Cluster'].unique()):
        cluster_data = df_filtered[df_filtered['Cluster'] == cluster]
        segment_name = cluster_segments[cluster]
        color = colors_map[cluster]
        ax.scatter(cluster_data['AnnualIncome'], cluster_data['TotalSpent'], 
                  s=150, alpha=0.6, label=f'Cluster {cluster}: {segment_name}',
                  color=color, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Annual Income (₹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Spent (₹)', fontsize=12, fontweight='bold')
    ax.set_title('Income vs Spending - Customer Clustering', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Detailed Cluster Statistics
    st.subheader("Detailed Cluster Statistics")
    for cluster in sorted(selected_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        segment = cluster_segments[cluster]
        
        with st.expander(f"📋 Cluster {cluster}: {segment} ({len(cluster_data)} customers)"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Customer Count", len(cluster_data))
            with col2:
                st.metric("Avg Income", f"₹{cluster_data['AnnualIncome'].mean():,.0f}")
            with col3:
                st.metric("Avg Spending", f"₹{cluster_data['TotalSpent'].mean():,.0f}")
            with col4:
                st.metric("Avg Monthly Purchases", f"{cluster_data['MonthlyPurchases'].mean():.1f}")
            
            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric("Avg App Usage", f"{cluster_data['AppTimeMinutes'].mean():.1f}m")
            with col6:
                st.metric("Avg Order Value", f"₹{cluster_data['AvgOrderValue'].mean():,.0f}")
            with col7:
                st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f}")
            
            # Additional stats
            st.write("**Demographics:**")
            gender_dist = cluster_data['Gender'].value_counts()
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if 'Male' in gender_dist.index:
                    st.write(f"Male: {gender_dist['Male']}")
            with col_b:
                if 'Female' in gender_dist.index:
                    st.write(f"Female: {gender_dist['Female']}")
            with col_c:
                if 'Other' in gender_dist.index:
                    st.write(f"Other: {gender_dist['Other']}")
            
            st.write("**Preferences:**")
            discount_dist = cluster_data['DiscountUsage'].value_counts()
            time_dist = cluster_data['PreferredShoppingTime'].value_counts()
            col_x, col_y = st.columns(2)
            with col_x:
                st.write("**Discount Usage:**")
                for idx, count in discount_dist.items():
                    st.write(f"  {idx}: {count}")
            with col_y:
                st.write("**Preferred Shopping Time:**")
                for idx, count in time_dist.items():
                    st.write(f"  {idx}: {count}")

# ============================================================================
# PAGE 3: CUSTOMER DETAILS
# ============================================================================
elif page == "👥 Customer Details":
    st.header("Customer Database")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_cluster = st.multiselect(
            "Filter by Cluster",
            options=sorted(df['Cluster'].unique()),
            default=sorted(df['Cluster'].unique()),
            format_func=lambda x: f"Cluster {x}: {cluster_segments[x]}"
        )
    
    with col2:
        min_spending = st.slider("Minimum Spending (₹)", 
                                int(df['TotalSpent'].min()), 
                                int(df['TotalSpent'].max()),
                                int(df['TotalSpent'].min()))
    
    with col3:
        min_income = st.slider("Minimum Income (₹)", 
                              int(df['AnnualIncome'].min()), 
                              int(df['AnnualIncome'].max()),
                              int(df['AnnualIncome'].min()))
    
    # Apply filters
    df_display = df[(df['Cluster'].isin(selected_cluster)) & 
                   (df['TotalSpent'] >= min_spending) &
                   (df['AnnualIncome'] >= min_income)].copy()
    
    # Add segment column for display
    df_display['Segment'] = df_display['Cluster'].map(cluster_segments)
    
    st.markdown(f"**Showing {len(df_display)} out of {len(df)} customers**")
    
    # Display data
    display_cols = ['Age', 'Gender', 'AnnualIncome', 'TotalSpent', 'AvgOrderValue', 
                   'MonthlyPurchases', 'AppTimeMinutes', 'Cluster', 'Segment']
    
    st.dataframe(df_display[display_cols], use_container_width=True, hide_index=True)
    
    # Download button
    csv = df_display[display_cols].to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="customer_data_filtered.csv",
        mime="text/csv"
    )

# ============================================================================
# PAGE 4: PREDICT CLUSTER
# ============================================================================
elif page == "🔮 Predict Cluster":
    st.header("Predict Cluster for New Customer")
    
    st.info("Enter customer details below to predict which cluster they belong to.")
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
        annual_income = st.number_input("Annual Income (₹)", min_value=20000, max_value=500000, value=100000)
        total_spent = st.number_input("Total Spent (₹)", min_value=0, max_value=100000, value=50000)
    
    with col2:
        avg_order_value = st.number_input("Avg Order Value (₹)", min_value=0, max_value=10000, value=500)
        monthly_purchases = st.number_input("Monthly Purchases", min_value=0, max_value=100, value=10)
        app_time = st.number_input("App Time (Minutes)", min_value=0, max_value=1000, value=100)
        discount_usage = st.selectbox("Discount Usage", options=['Yes', 'No', 'Sometimes'])
        shopping_time = st.selectbox("Preferred Shopping Time", 
                                    options=['Morning', 'Afternoon', 'Evening', 'Night'])
    
    if st.button("Predict Cluster", type="primary"):
        try:
            # Encode categorical variables using the same encoders
            # We'll use the values from the training data to create encoders
            gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
            discount_mapping = {'No': 0, 'Sometimes': 1, 'Yes': 2}
            time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
            
            # Create feature vector
            features = [
                age,
                gender_mapping[gender],
                annual_income,
                total_spent,
                avg_order_value,
                monthly_purchases,
                discount_mapping[discount_usage],
                app_time,
                time_mapping[shopping_time]
            ]
            
            # Scale features
            X_scaled = scaler.transform([features])
            
            # Predict cluster
            predicted_cluster = kmeans.predict(X_scaled)[0]
            predicted_segment = cluster_segments[predicted_cluster]
            
            st.success("✅ Prediction Successful")
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Cluster", predicted_cluster)
            with col2:
                st.metric("Customer Segment", predicted_segment)
            with col3:
                # Get cluster info
                cluster_avg_spend = df[df['Cluster'] == predicted_cluster]['TotalSpent'].mean()
                st.metric("Cluster Avg Spend", f"₹{cluster_avg_spend:,.0f}")
            
            st.markdown("---")
            st.subheader("Cluster Profile Comparison")
            
            # Compare with cluster average
            comparison_data = []
            cluster_data = df[df['Cluster'] == predicted_cluster]
            
            metrics = {
                'Annual Income': annual_income,
                'Total Spent': total_spent,
                'Monthly Purchases': monthly_purchases,
                'App Time (min)': app_time,
                'Avg Order Value': avg_order_value
            }
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Your Input:**")
                for metric, value in metrics.items():
                    if 'Income' in metric or 'Spent' in metric or 'Order Value' in metric:
                        st.write(f"  {metric}: ₹{value:,.0f}")
                    else:
                        st.write(f"  {metric}: {value}")
            
            with col_b:
                st.write(f"**{predicted_segment} (Cluster {predicted_cluster}) Average:**")
                avg_metrics = {
                    'Annual Income': f"₹{cluster_data['AnnualIncome'].mean():,.0f}",
                    'Total Spent': f"₹{cluster_data['TotalSpent'].mean():,.0f}",
                    'Monthly Purchases': f"{cluster_data['MonthlyPurchases'].mean():.1f}",
                    'App Time (min)': f"{cluster_data['AppTimeMinutes'].mean():.1f}",
                    'Avg Order Value': f"₹{cluster_data['AvgOrderValue'].mean():,.0f}"
                }
                for metric, value in avg_metrics.items():
                    st.write(f"  {metric}: {value}")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Customer Clustering Analysis Dashboard** | Powered by Streamlit")
