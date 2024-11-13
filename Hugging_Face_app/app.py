
import pickle
import re
import gradio as gr

# Load stopwords
# stop_words = set(stopwords.words("english"))
stop_words = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my",
    "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
]
tag_pattern = re.compile(r'<[^>]+>')

# Preprocessing function
def preprocess_text(text):
    text = tag_pattern.sub('', text)
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load TF-IDF vectorizer and logistic regression model
with open(r'tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)  # Load TF-IDF vectorizer
with open(r'model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)  # Load logistic regression model

# Prediction function for logistic regression model with TF-IDF
def predict_sentiment_logistic(input_text):
    preprocessed_text = preprocess_text(input_text)
    # Convert text to TF-IDF feature vector
    feature_vector = tfidf_vectorizer.transform([preprocessed_text])  # Transform text using TF-IDF
    prediction = logistic_model.predict(feature_vector)[0]
    confidence = logistic_model.predict_proba(feature_vector)[0][1]  # Probability of the positive class
    sentiment_label = "POSITIVE" if prediction == 1 else "NEGATIVE"
    sentiment_score = round(confidence * 100, 2) if prediction == 1 else round((1 - confidence) * 100, 2)
    
    # HTML output for styling
    output_html = f"""
    <div style="font-size: 20px; text-align: center;">
        <strong style="font-size: 30px; color: {'#28a745' if prediction == 1 else '#dc3545'};">{sentiment_label}</strong>
        <br>
        <span style="font-size: 18px; color: #6c757d;">{sentiment_label} ....... {sentiment_score}%</span>
        <div style="width: 100%; background-color: #e9ecef; border-radius: 8px; overflow: hidden; margin-top: 10px;">
            <div style="width: {sentiment_score}%; height: 10px; background-color: {'#28a745' if prediction == 1 else '#dc3545'};"></div>
        </div>
    </div>
    """
    return output_html

# Gradio interface
examples = [
    "The product exceeded my expectations and works perfectly!",
    "I'm very disappointed",
    "I absolutely love this restaurant! The food and atmosphere are incredible.",
    "This movie was a waste of time. The plot was confusing and the acting was bad."
]

interface = gr.Interface(
    fn=predict_sentiment_logistic,
    inputs=gr.Textbox(label="Customer Review"),
    outputs=gr.HTML(label="Sentiment Level"),
    examples=examples,
    title="Customer Sentiment Analysis",
    description="Analyze the sentiment of a review as Positive or Negative with a confidence score.",
    theme="default",
)

interface.launch()


