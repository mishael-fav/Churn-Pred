import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .churn-alert {
        background-color: #fee2e2;
        border-left: 5px solid #dc2626;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .no-churn-alert {
        background-color: #d1fae5;
        border-left: 5px solid #059669;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Expected features (must match training data)
EXPECTED_FEATURES = [
    'NumOrders', 'PurchaseDays', 'TotalRevenue', 'AvgOrderValue',
    'TotalQuantity', 'NumLineItems', 'CustomerLifespanDays',
    'PurchaseFrequency', 'AvgItemsPerOrder'
]

@st.cache_resource(show_spinner=False)
def load_model():
    """Load ML model with comprehensive error handling"""
    try:
        model_path = Path("churn_model.pkl")

        if not model_path.exists():
            st.error("❌ **Model file not found!**")
            st.info("Please ensure 'churn_model.pkl' is in the same directory as this app.")
            st.stop()

        model = joblib.load(model_path)
        return model

    except Exception as e:
        st.error(f"❌ **Error loading model:** {str(e)}")
        st.info("Please check if the model file is compatible with the current environment.")
        st.stop()

def validate_batch_data(df):
    """Validate uploaded CSV data"""
    errors = []

    # Check for required columns
    missing_cols = set(EXPECTED_FEATURES) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {', '.join(missing_cols)}")

    # Check for extra columns (warning only)
    extra_cols = set(df.columns) - set(EXPECTED_FEATURES)

    if errors:
        return False, errors, extra_cols

    # Check for non-numeric values
    try:
        df[EXPECTED_FEATURES].astype(float)
    except ValueError as e:
        errors.append(f"Non-numeric values detected. All features must be numeric.")
        return False, errors, extra_cols

    # Check for null values
    null_counts = df[EXPECTED_FEATURES].isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].to_dict()
        errors.append(f"Missing values found: {null_cols}")
        return False, errors, extra_cols

    # Check for negative values (business logic)
    negative_cols = []
    for col in EXPECTED_FEATURES:
        if (df[col] < 0).any():
            negative_cols.append(col)

    if negative_cols:
        errors.append(f"Negative values found in: {', '.join(negative_cols)}")
        return False, errors, extra_cols

    return True, [], extra_cols

def display_single_prediction(prediction, probability=None):
    """Display prediction result with styling"""
    if prediction == 1:
        st.markdown("""
            <div class="churn-alert">
                <h2>🚨 HIGH CHURN RISK</h2>
                <p style="font-size: 1.1rem; margin-top: 0.5rem;">
                    This customer is predicted to <strong>CHURN</strong>.
                    Consider implementing retention strategies immediately.
                </p>
            </div>
        """, unsafe_allow_html=True)

        if probability is not None:
            st.error(f"**Churn Probability:** {probability:.1%}")
            st.progress(float(probability))
    else:
        st.markdown("""
            <div class="no-churn-alert">
                <h2>✅ LOW CHURN RISK</h2>
                <p style="font-size: 1.1rem; margin-top: 0.5rem;">
                    This customer is predicted to <strong>STAY ACTIVE</strong>.
                    Continue engagement strategies to maintain loyalty.
                </p>
            </div>
        """, unsafe_allow_html=True)

        if probability is not None:
            st.success(f"**Retention Probability:** {1-probability:.1%}")
            st.progress(float(1-probability))

def create_feature_dataframe(values):
    """Create properly formatted DataFrame for prediction"""
    return pd.DataFrame([values], columns=EXPECTED_FEATURES)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction System</h1>', unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading ML model..."):
        model = load_model()

    st.success("✅ Model loaded successfully!")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/analytics.png", width=100)
        st.title("⚙️ Settings")

        mode = st.radio(
            "**Choose Input Mode:**",
            ["📝 Single Customer", "📊 Batch Upload"],
            index=0
        )

        st.markdown("---")

        # Information panel
        with st.expander("ℹ️ About This App"):
            st.markdown("""
                This application predicts customer churn using machine learning.

                **Features:**
                - Single customer prediction
                - Batch CSV upload
                - Churn probability scores
                - Downloadable results
            """)

        with st.expander("📋 Feature Definitions"):
            st.markdown("""
                - **NumOrders**: Total orders placed
                - **PurchaseDays**: Unique days with purchases
                - **TotalRevenue**: Cumulative spending ($)
                - **AvgOrderValue**: Average order value ($)
                - **TotalQuantity**: Total items bought
                - **NumLineItems**: Total transaction lines
                - **CustomerLifespanDays**: Days since first purchase
                - **PurchaseFrequency**: Orders per active day
                - **AvgItemsPerOrder**: Items per transaction
            """)

        st.markdown("---")
        st.caption("💡 Powered by Machine Learning")

    # Main content area
    if "Single Customer" in mode:
        st.header("📝 Single Customer Prediction")
        st.markdown("Enter the customer metrics below to predict churn risk.")

        # Create input form
        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                NumOrders = st.number_input(
                    "Number of Orders",
                    min_value=0,
                    value=5,
                    step=1,
                    help="Total number of orders placed by customer"
                )
                PurchaseDays = st.number_input(
                    "Purchase Days",
                    min_value=0,
                    value=3,
                    step=1,
                    help="Number of unique days customer made purchases"
                )
                TotalRevenue = st.number_input(
                    "Total Revenue ($)",
                    min_value=0.0,
                    value=250.0,
                    step=10.0,
                    format="%.2f",
                    help="Total amount spent by customer"
                )

            with col2:
                AvgOrderValue = st.number_input(
                    "Avg Order Value ($)",
                    min_value=0.0,
                    value=50.0,
                    step=5.0,
                    format="%.2f",
                    help="Average value per order"
                )
                TotalQuantity = st.number_input(
                    "Total Quantity",
                    min_value=0,
                    value=25,
                    step=1,
                    help="Total number of items purchased"
                )
                NumLineItems = st.number_input(
                    "Number of Line Items",
                    min_value=0,
                    value=10,
                    step=1,
                    help="Total transaction lines across all orders"
                )

            with col3:
                CustomerLifespanDays = st.number_input(
                    "Customer Lifespan (days)",
                    min_value=0,
                    value=90,
                    step=1,
                    help="Days from first to last purchase"
                )
                PurchaseFrequency = st.number_input(
                    "Purchase Frequency",
                    min_value=0.0,
                    value=0.05,
                    step=0.01,
                    format="%.4f",
                    help="Average orders per day active"
                )
                AvgItemsPerOrder = st.number_input(
                    "Avg Items per Order",
                    min_value=0.0,
                    value=2.5,
                    step=0.1,
                    format="%.2f",
                    help="Average number of items per order"
                )

            submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

        if submitted:
            # Create feature array
            features_df = create_feature_dataframe([
                NumOrders, PurchaseDays, TotalRevenue, AvgOrderValue,
                TotalQuantity, NumLineItems, CustomerLifespanDays,
                PurchaseFrequency, AvgItemsPerOrder
            ])

            with st.spinner("Analyzing customer behavior..."):
                try:
                    # Make prediction
                    prediction = model.predict(features_df)[0]

                    # Get probability if available
                    probability = None
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_df)[0]
                        probability = proba[1]  # Probability of churn (class 1)

                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    display_single_prediction(prediction, probability)

                    # Show input summary
                    with st.expander("📋 View Input Data"):
                        st.dataframe(
                            features_df.T.rename(columns={0: "Value"}),
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"❌ **Prediction Error:** {str(e)}")
                    st.info("Please check your input values and try again.")

    else:  # Batch Upload mode
        st.header("📊 Batch Prediction from CSV")
        st.markdown("Upload a CSV file containing customer data for bulk predictions.")

        # Template download section
        col_temp1, col_temp2 = st.columns([2, 1])

        with col_temp1:
            st.info("💡 **Tip:** Download the template CSV to see the required format.")

        with col_temp2:
            # Create template
            template_data = {col: [0.0] for col in EXPECTED_FEATURES}
            template_df = pd.DataFrame(template_data)

            st.download_button(
                label="⬇️ Download Template",
                data=template_df.to_csv(index=False),
                file_name="churn_prediction_template.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload CSV with customer metrics"
        )

        if uploaded_file is not None:
            try:
                # Read uploaded file
                data = pd.read_csv(uploaded_file)

                st.success(f"✅ File uploaded: **{uploaded_file.name}** ({len(data)} records)")

                # Validate data
                is_valid, errors, extra_cols = validate_batch_data(data)

                if not is_valid:
                    st.error("❌ **Validation Failed**")
                    for error in errors:
                        st.warning(f"⚠️ {error}")

                    st.markdown("**Preview of uploaded data:**")
                    st.dataframe(data.head(10), use_container_width=True)
                    st.stop()

                # Show warnings for extra columns
                if extra_cols:
                    st.warning(f"⚠️ Extra columns detected (will be ignored): {', '.join(extra_cols)}")

                # Preview data
                with st.expander("👀 Preview Data (First 10 rows)", expanded=True):
                    st.dataframe(data.head(10), use_container_width=True)

                # Predict button
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=False):
                    with st.spinner("Processing predictions... This may take a moment."):
                        try:
                            # Extract features in correct order
                            features_data = data[EXPECTED_FEATURES].copy()

                            # Make predictions
                            predictions = model.predict(features_data)

                            # Add results to dataframe
                            result_df = data.copy()
                            result_df['ChurnPrediction'] = predictions
                            result_df['ChurnLabel'] = result_df['ChurnPrediction'].map({
                                0: 'No Churn',
                                1: 'Churn'
                            })

                            # Add probabilities if available
                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(features_data)
                                result_df['ChurnProbability'] = probabilities[:, 1]
                                result_df['ChurnProbability'] = result_df['ChurnProbability'].round(4)

                            st.success("✅ **Predictions completed successfully!**")

                            # Display metrics
                            st.markdown("### 📈 Summary Statistics")

                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                            total_customers = len(result_df)
                            churn_count = (predictions == 1).sum()
                            no_churn_count = (predictions == 0).sum()
                            churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0

                            with col_m1:
                                st.metric("Total Customers", f"{total_customers:,}")
                            with col_m2:
                                st.metric("Predicted Churns", f"{churn_count:,}", delta=f"{churn_rate:.1f}%", delta_color="inverse")
                            with col_m3:
                                st.metric("Active Customers", f"{no_churn_count:,}")
                            with col_m4:
                                st.metric("Churn Rate", f"{churn_rate:.1f}%", delta_color="off")

                            # Display results
                            st.markdown("### 📋 Prediction Results")
                            st.dataframe(
                                result_df,
                                use_container_width=True,
                                height=400
                            )

                            # Download section
                            st.markdown("### 💾 Download Results")

                            col_d1, col_d2 = st.columns(2)

                            with col_d1:
                                csv_data = result_df.to_csv(index=False)
                                st.download_button(
                                    label="⬇️ Download Full Results (CSV)",
                                    data=csv_data,
                                    file_name="churn_predictions_full.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                            with col_d2:
                                # Download only churned customers
                                churned_df = result_df[result_df['ChurnPrediction'] == 1]
                                csv_churned = churned_df.to_csv(index=False)
                                st.download_button(
                                    label="⬇️ Download Churned Only (CSV)",
                                    data=csv_churned,
                                    file_name="churn_predictions_churned.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    disabled=(churn_count == 0)
                                )

                        except Exception as e:
                            st.error(f"❌ **Prediction Error:** {str(e)}")
                            st.exception(e)

            except Exception as e:
                st.error(f"❌ **Error reading CSV file:** {str(e)}")
                st.info("Please ensure the file is a valid CSV format.")

if __name__ == "__main__":
    main()