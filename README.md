# Research Abstract Classification

### Machine Learning-Based Classification of Research Abstracts

A machine learning project that automatically classifies research paper abstracts into three categories:

- **Science**
- **Business**
- **Social Science**

## Project Overview

The objective of this project is to develop a text classification system that can identify the research domain of an academic abstract using Natural Language Processing (NLP) and machine learning techniques.

The project compares multiple classification algorithms and selects the best-performing model for deployment through a Streamlit web application.

## Dataset

The dataset was collected using OpenAlex metadata through subject-related search queries.

| Category | Number of Abstracts |
|---|---:|
| Science | 3,000 |
| Business | 3,000 |
| Social Science | 2,983 |
| **Total** | **8,983** |

### Dataset Preparation

The dataset was checked for:

- Missing or empty abstracts
- Duplicate abstracts
- Duplicate OpenAlex IDs
- Missing labels
- Missing values

The final dataset contains 8,983 usable research abstracts.

## Methodology

The project follows these major steps:

1. Data collection using OpenAlex.
2. Data cleaning and preprocessing.
3. Text feature extraction using TF-IDF.
4. Training and evaluation of multiple machine learning models.
5. Model comparison using test-set performance.
6. Selection of the best-performing model.
7. Deployment using Streamlit.

The dataset was divided into training and testing sets using an 80:20 stratified split with `random_state=42`.

TF-IDF vectorization was performed within the machine learning pipelines to avoid fitting the vectorizer on test data.

## Models Compared

Seven machine learning configurations were evaluated.

| Model | Test Accuracy |
|---|---:|
| SVM (Word + Character TF-IDF) | **91.93%** |
| SVM (Word TF-IDF) | 91.82% |
| Logistic Regression | 91.21% |
| K-Nearest Neighbors | 90.37% |
| Naive Bayes | 90.15% |
| Random Forest | 87.76% |
| Decision Tree | 74.62% |

## Best Model

The selected model is **Support Vector Machine (SVM) using Word and Character TF-IDF features**.

It achieved the highest test accuracy and was selected based on its macro F1-score.

### Classification Performance

| Metric | Score |
|---|---:|
| Accuracy | 91.93% |
| Macro Precision | 91.97% |
| Macro Recall | 91.93% |
| Macro F1-score | 91.94% |

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Business | 0.94 | 0.92 | 0.93 |
| Science | 0.95 | 0.94 | 0.95 |
| Social Science | 0.88 | 0.89 | 0.88 |

## Web Application

The trained model is deployed using Streamlit.

**Live Demo:**  
https://research-abstract-classifier-kpxy7pyqneancqbnze5d9j.streamlit.app/

The application supports:

- Classification of individual research abstracts.
- Batch classification using CSV files.
- Display of predicted research categories.
- Downloading batch prediction results as a CSV file.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/amandam8842-hue/research-abstract-classifier.git
```

### 2. Navigate to the project directory

```bash
cd research-abstract-classifier
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## CSV Input Format

For batch predictions, upload a CSV file containing a column named:

```text
abstract
```

The application generates predictions for each abstract and allows the results to be downloaded.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Support Vector Machine
- Joblib
- Streamlit
- OpenAlex API

## Project Structure

```text
research-abstract-classifier/
│
├── app.py
├── best_model.joblib
├── requirements.txt
├── README.md
└── notebook/
    └── research_abstract_classification.ipynb
```

*The exact notebook folder and filename may differ depending on the repository structure.*

## Limitations

- The classification performance depends on the quality and coverage of the collected abstracts.
- The labels were associated with subject-related OpenAlex search queries and were not independently verified by human annotators.
- Therefore, the reported evaluation measures agreement with the working dataset labels and may not fully represent performance on independently labelled research abstracts.

## Conclusion

This project demonstrates how NLP and machine learning can be used to classify academic research abstracts into Science, Business, and Social Science categories.

Among the seven evaluated configurations, SVM with Word and Character TF-IDF achieved the best test accuracy of **91.93%** and was selected for deployment.

## Author

**Aman Dam**

**GitHub Repository:**  
https://github.com/amandam8842-hue/research-abstract-classifier

**Live Application:**  
https://research-abstract-classifier-kpxy7pyqneancqbnze5d9j.streamlit.app/
