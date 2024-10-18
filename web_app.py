import streamlit as st
import nltk
from nltk import ngrams
from collections import defaultdict
import numpy as np
import requests
from bs4 import BeautifulSoup
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Download necessary NLTK data files
nltk.download('punkt')

def tokenize_code(code):
    return nltk.word_tokenize(code)

def build_ngram_model(corpus, n=3):
    ngram_freqs = defaultdict(int)
    for text in corpus:
        tokens = tokenize_code(text)
        for ngram in ngrams(tokens, n):
            ngram_freqs[ngram] += 1
    return ngram_freqs

def calculate_ngram_likelihood(ngram_freqs, code_sample, n=3):
    tokens = tokenize_code(code_sample)
    ngram_count = len(list(ngrams(tokens, n)))
    likelihood = 0
    for ngram in ngrams(tokens, n):
        likelihood += np.log(ngram_freqs.get(ngram, 1))
    return np.exp(likelihood / ngram_count)

def calculate_perplexity(ngram_freqs, code_sample, n=3):
    likelihood = calculate_ngram_likelihood(ngram_freqs, code_sample, n)
    return np.exp(-likelihood)

def check_web_copy(code_sample):
    search_url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(" ".join(code_sample.split()[:10]))
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True) if 'url?q=' in a['href']]
    return [f"<a href='{urllib.parse.unquote(link.split('url?q=')[1].split('&')[0])}' target='_blank'>{urllib.parse.unquote(link.split('url?q=')[1].split('&')[0])}</a>" for link in links[:3]]

def calculate_similarity(corpus, code_sample):
    vectorizer = TfidfVectorizer().fit_transform(corpus + [code_sample])
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1])
    return cosine_similarities.flatten()

def classify_code_ml(code_sample, model, vectorizer):
    sample_vector = vectorizer.transform([code_sample])
    prediction = model.predict(sample_vector)
    return "Likely AI-generated" if prediction == 1 else "Likely Human-written"

def extract_features(code_sample):
    lines = code_sample.strip().split('\n')
    num_lines = len(lines)
    num_tokens = len(tokenize_code(code_sample))
    comment_density = sum(1 for line in lines if '#' in line) / num_lines
    return num_lines, num_tokens, comment_density

def train_classifier(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, vectorizer

# Streamlit App
st.set_page_config(page_title="Code Analysis", layout="wide")
st.title("🔍 Code Analysis and Plagiarism Detection")

# Input box for code
st.sidebar.header("Input Code")
code_sample = st.sidebar.text_area("Enter your code here:")

if st.sidebar.button("Analyze Code"):
    # Example corpus of human-written code samples
    human_corpus = ["def example(): pass", "def another_example(): return 42"]  # Replace with real corpus
    labels = ['human'] * len(human_corpus)

    # Train classifier
    model, vectorizer = train_classifier(human_corpus, labels)

    # Build n-gram model
    ngram_freqs = build_ngram_model(human_corpus, n=3)

    # Classify code
    likelihood = calculate_ngram_likelihood(ngram_freqs, code_sample)
    perplexity = calculate_perplexity(ngram_freqs, code_sample)
    classification = classify_code_ml(code_sample, model, vectorizer)

    # Check if code is copied from the web
    sources = check_web_copy(code_sample)

    # Calculate similarity with corpus
    similarities = calculate_similarity(human_corpus, code_sample)

    # Display analysis
    st.subheader("Analysis Results")
    st.markdown(f"**N-gram Likelihood:** {likelihood:.4f}")
    st.markdown(f"**Perplexity:** {perplexity:.4f} (Lower values indicate more predictable code)")
    st.markdown(f"**Classification:** {classification}")

    # Plotly Visualizations
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(name='Metrics', x=["Likelihood", "Perplexity"], y=[likelihood, perplexity])
        ])
        fig.update_layout(title="Likelihood and Perplexity", xaxis_title="Metric", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_sim = px.scatter(x=[f"Sample {i+1}" for i in range(len(similarities))], y=similarities, labels={'x': 'Sample', 'y': 'Similarity Score'}, title="Similarity Scores with Corpus")
        st.plotly_chart(fig_sim, use_container_width=True)

    # Display clickable links
    st.subheader("Potential Sources")
    for source in sources:
        st.markdown(source, unsafe_allow_html=True)

    # Additional Visualization
    st.subheader("Advanced Visualizations")
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[likelihood, perplexity, len(code_sample.split()), len(set(code_sample.split()))],
        theta=['Likelihood', 'Perplexity', 'Token Count', 'Unique Tokens'],
        fill='toself'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, title="Code Feature Radar Chart")
    st.plotly_chart(radar_fig, use_container_width=True)
