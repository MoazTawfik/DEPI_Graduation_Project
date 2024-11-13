# DEPI_Graduation_Project

![My Image](images/Sentiment-analysis.svg)

# Overview:
This project analyzes customer feedback to classify sentiment as either positive or negative using Natural Language Processing (NLP) techniques. By leveraging a combination of traditional machine learning methods and deep learning models, the project aims to uncover sentiment trends, providing valuable insights into customer behavior and preferences to help businesses make more informed decisions.

Key Objectives:

* Perform sentiment classification using diverse datasets.
* Apply machine learning and deep learning models, including Bert and DistilBERT.

# Web Application:
* [HuggingFce](https://huggingface.co/spaces/MoazTawfik/Customer_Sentiment_Analysis): A live demo of the sentiment analysis model hosted on HuggingFace Spaces.
* [Streamlit](https://depisentimentanalysisapp-11-6-2024.streamlit.app/) :Interactive web application built with Streamlit for real-time sentiment analysis
  
# Datasets Used
For the sentiment analysis task, we utilized well-known datasets to ensure comprehensive model training and evaluation:

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) : 1.6M Labeled tweet.

[Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews/data): Amazon product reviews with sentiment labels.

[IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) : 50k movie reviews labeled as positive or negative.

[Yelp Datset](https://huggingface.co/datasets/Yelp/yelp_review_full) : offers business and review data for analyzing customer sentiment and behavior.


# Key Features
# 1-Data Preprocessing
We prepare the data for modeling by:

* Text Cleaning: Removing noise such as URLs, stop words, and punctuation.
* Normalization: Applying tokenization, stemming, and lemmatization.
* Feature Extraction: Transforming text into numerical representations using onehotencoding , TF-IDF, CountVectorizer, and GloVe embeddings.

# 2-Modeling
We experimented with a variety of models to optimize performance:

* Traditional Models: Logistic Regression, Random Forest,KNN  and XGboost.
* Deep Learning Models:
  * GRU ,Bidirectional
  * LSTM (Long Short-Term Memory): Ideal for capturing sequential patterns in text data.
  * DistilBERT: A streamlined, faster version of BERT, enhancing efficiency in sentiment classification.
  * RoBerta,Bert.
  
# 3-Results
Achieved: 87% accuracy with RoBERTa.

Achieved: 90% accuracy with Logistic Regression.

# Technologies
* NLP Models: BERT, RoBERTa, DistilBERT.
  
* python Libraries: NumPy, Pandas, Scikit-learn, TensorFlow, NLTK,seaborn  and Matplotlib.
 
* Deployment Platforms: Microsoft Azure, with MLflow for MLOps integration, Hugging Face, and Streamlit.

# This project was developed as a graduation project for Microsoft Machine Learning Program within DEPI.
![DEPI](images/DEPI.png)

# Contact
If you have any questions or would like to contribute, don’t hesitate to connect with me via [LinkedIn](https://www.linkedin.com/in/moaz-tawfik-b276502a7) or GitHub. I'm always open to collaboration and discussions!


