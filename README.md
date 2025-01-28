### Mental Health Status Dataset for AI and Sentiment Analysis 🚀  

Welcome to the **Mental Health Status Dataset** repository! This repository contains a meticulously curated dataset designed to support AI applications, sentiment analysis, and research in mental health.  

---

## 🗂️ About the Dataset  
This dataset is a compilation of mental health statuses derived from various textual statements. It has been cleaned and organized to serve as a valuable resource for:  

- ✅ **AI Chatbot Development**: For building mental health support systems.  
- ✅ **Sentiment Analysis**: To understand and analyze mental health trends.  
- ✅ **Research**: Academic studies on mental health patterns and behavior.  

### 🌟 Key Features:  

| **Column Name**        | **Description**                                      |  
|-------------------------|------------------------------------------------------|  
| `unique_id`            | A unique identifier for each entry.                 |  
| `Statement`            | Textual data or post (e.g., social media content).   |  
| `Mental Health Status` | The tagged mental health status (one of 7 classes).  |  

### 🧠 **Mental Health Status Tags**  
The dataset includes seven categories of mental health statuses:  
1. **Normal**  
2. **Depression**  
3. **Suicidal**  
4. **Anxiety**  
5. **Stress**  
6. **Bi-Polar**  
7. **Personality Disorder**  

---

## 📈 Applications  
This dataset is versatile and can be applied in:  
- **Training Machine Learning Models** to predict mental health conditions.  
- **Building Intelligent Chatbots** for mental health support.  
- **Academic Research**: Understanding mental health patterns through sentiment analysis.  

---

## 📚 Data Sources  
The dataset is an aggregation of publicly available data from the following Kaggle sources:  
- **3k Conversations Dataset for Chatbot**  
- **Depression Reddit Cleaned**  
- **Human Stress Prediction**  
- **Predicting Anxiety in Mental Health Data**  
- **Mental Health Dataset Bipolar**  
- **Reddit Mental Health Data**  
- **Students Anxiety and Depression Dataset**  
- **Suicidal Mental Health Dataset**  
- **Suicidal Tweet Detection Dataset**  

Huge thanks to the original contributors for their invaluable work! 🙌  

---

## 🚀 Getting Started  

### 🔧 **Installation and Setup**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/username/mental-health-status-dataset.git  
   cd mental-health-status-dataset  
   ```  
2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

### 📊 **Exploratory Data Analysis (EDA)**  
- Analyze the distribution of mental health statuses.  
- Visualize textual patterns and word clouds for insights.  

### 🔧 **Data Preprocessing**  
- Text cleaning: Tokenization, stemming, and stopword removal.  
- Feature engineering for NLP-based tasks.  

### 🤖 **Model Training**  
- Train machine learning models like **Logistic Regression**, **SVM**, and **Random Forest**.  
- Use deep learning models such as **RNN**, **LSTM**, and **Transformers** for advanced predictions.  

### 🚀 **Deployment**  
- Export trained models for deployment in real-world applications.  
- Deploy models using **Flask** or **FastAPI** for AI chatbot integration.  

---

## 📝 Code Implementation  
Explore the full implementation, including:  
1. **EDA**: Insights and visualizations 📊  
2. **Data Preprocessing**: Clean and prepare data 🔧  
3. **Model Development**: Train machine learning and deep learning models 🤖  
4. **Evaluation**: Model performance metrics (accuracy, precision, recall, F1-score) 📈  
5. **Deployment**: APIs for real-world integration 🚀  

---

## 📄 Acknowledgments  
This dataset was aggregated from publicly available sources on Kaggle. A big thank you to the original dataset creators for their contributions to the field of mental health.  

---

## 💬 Contributing  
We welcome contributions! Feel free to open issues or submit pull requests to enhance the dataset or improve the codebase.  

---

## 📧 Contact  
For any questions or collaborations, feel free to reach out:  
📩 **Email**: arifmiahcse@gmail.com 
🌐 **LinkedIn**: [Your LinkedIn Profile](ww.linkedin.com/in/arif-miah-8751bb217)  

---

## 🌟 Show Your Support  
If you find this dataset helpful, please ⭐ the repository and share it with others in your network. Together, we can make a difference in mental health research! 💙  



---

### Exploratory Data Analysis (EDA) Code

Here’s an example of EDA for the dataset:  

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv("mental_health_status_dataset.csv")

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Preview the dataset
print("\nDataset Head:")
print(df.head())

# Plot the distribution of mental health statuses
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Mental Health Status", palette="viridis")
plt.title("Distribution of Mental Health Statuses", fontsize=16)
plt.xlabel("Mental Health Status")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Generate a WordCloud for textual data (statements)
text = " ".join(df['Statement'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Statements", fontsize=16)
plt.show()
```

---

### Model Training Code

This code trains a basic Logistic Regression model to classify mental health statuses.

```python
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# Clean the data (remove missing values)
df = df.dropna(subset=["Statement", "Mental Health Status"])

# Split the data into features and target
X = df["Statement"]
y = df["Mental Health Status"]

# Encode target labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```

---



  
