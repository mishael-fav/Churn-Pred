# 🎯 Customer Churn Prediction System

An intelligent machine learning application designed to predict customer churn by analyzing purchase behavior patterns. This tool empowers businesses to identify at-risk customers and take proactive steps to improve retention.

---

## 🌟 Features

* **Single Customer Prediction** → Enter individual metrics for instant churn risk assessment
* **Batch Prediction** → Upload CSV files for bulk analysis across multiple customers
* **Probability Scores** → View confidence levels for each prediction
* **Visual Analytics** → Monitor churn trends and retention statistics
* **Export Results** → Download churn predictions as CSV files for business use

---

## 🛠️ Technology Stack

* **Framework**: Streamlit
* **Machine Learning**: Scikit-learn
* **Data Processing**: Pandas, NumPy
* **Model**: Trained on the Online Retail Dataset
* **Deployment**: Streamlit Cloud & Hugging Face Spaces

---

## 📊 Model Features

The churn model uses 9 engineered customer behavior metrics:

1. **NumOrders** → Total number of orders placed
2. **PurchaseDays** → Number of unique days with purchases
3. **TotalRevenue** → Cumulative customer spending
4. **AvgOrderValue** → Average value per order
5. **TotalQuantity** → Total items purchased
6. **NumLineItems** → Number of transaction line items
7. **CustomerLifespanDays** → Days between first and last purchase
8. **PurchaseFrequency** → Average orders per active day
9. **AvgItemsPerOrder** → Average items per transaction

---

## 🚀 Live Demo

* **Streamlit Cloud**: [Your App URL]
* **Hugging Face Spaces**: [Your Space URL]

---

## 💻 Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run churn_app.py
```

---

## 📈 Usage

### 🔹 Single Prediction

1. Select **"Single Entry"** mode
2. Input customer metrics
3. Click **Predict Churn**
4. View churn prediction and probability score

### 🔹 Batch Prediction

1. Select **"Batch Upload"** mode
2. Prepare a CSV file with customer data
3. Upload and generate churn predictions
4. Download results as a CSV file

---

## 📑 Sample CSV Templates

### ✅ Single Entry Example

```csv
NumOrders,PurchaseDays,TotalRevenue,AvgOrderValue,TotalQuantity,NumLineItems,CustomerLifespanDays,PurchaseFrequency,AvgItemsPerOrder
5,3,250.00,50.00,12,8,120,0.04,1.50
```

### ✅ Batch Upload Example

```csv
NumOrders,PurchaseDays,TotalRevenue,AvgOrderValue,TotalQuantity,NumLineItems,CustomerLifespanDays,PurchaseFrequency,AvgItemsPerOrder
5,3,250.00,50.00,12,8,120,0.04,1.50
10,7,750.00,75.00,30,20,240,0.05,1.80
2,1,80.00,40.00,4,3,30,0.06,1.33
```

---

## 📁 Project Structure

```
churn-predictor/
├── churn_app.py        # Main Streamlit application
├── churn_model.pkl     # Trained ML model
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## 🎓 Model Training

* **Dataset**: Online Retail Dataset (UCI Machine Learning Repository)
* **Algorithm**: Random Forest Classifier
* **Evaluation**: Accuracy, Precision, Recall, and AUC metrics
* **Features**: 9 engineered customer behavior metrics

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

* **Your Name**
* GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
* LinkedIn: [Your LinkedIn Profile]
* Portfolio: [Your Website]

---

## 🙏 Acknowledgments

* **Dataset**: UCI Machine Learning Repository
* **Framework**: Streamlit
* **Hosting**: Streamlit Cloud & Hugging Face Spaces
