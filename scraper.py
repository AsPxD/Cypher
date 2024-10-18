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
from bagofwords import bagofwords

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
    return [f"<a href='{urllib.parse.unquote(link.split('url?q=')[1].split('&')[0])}'>{link}</a>" for link in links[:3]]

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

def display_analysis(code_sample, likelihood, perplexity, classification, sources, similarities):
    num_lines, num_tokens, comment_density = extract_features(code_sample)
    
    print("\n--- Detailed Analysis ---")
    print(f"Code Sample:\n{code_sample.strip()}\n")
    print(f"N-gram Likelihood: {likelihood:.4f}")
    print(f"Perplexity: {perplexity:.4f} (Lower values indicate more predictable code)")
    print(f"Classification: {classification}")
    print(f"Lines of Code: {num_lines}")
    print(f"Number of Tokens: {num_tokens}")
    print(f"Comment Density: {comment_density:.2f}")
    print("Potential Sources:")
    for idx, source in enumerate(sources, 1):
        print(f"{idx}. {source}")
    print("Similarity Scores with Corpus:")
    for idx, similarity in enumerate(similarities, 1):
        print(f"Sample {idx}: {similarity:.4f}")
    print("\n--- Visual Summary ---")
    print("Perplexity Chart:")
    print("[" + "#" * int(perplexity * 10) + "-" * (50 - int(perplexity * 10)) + "]")

def train_classifier(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, vectorizer

def main():
    # Example corpus of human-written code samples
    human_corpus = bagofwords
    labels = ['human'] * len(human_corpus)  # Assuming all samples are human-written for now

    # Train classifier
    model, vectorizer = train_classifier(human_corpus, labels)

    # Build n-gram model
    ngram_freqs = build_ngram_model(human_corpus, n=3)

    # Code sample to classify
    code_sample = """
    def fourSum(nums, target):
    nums.sort()
    result = []
    
    for i in range(len(nums) - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        for j in range(i + 1, len(nums) - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            
            left, right = j + 1, len(nums) - 1
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                
                if current_sum < target:
                    left += 1
                elif current_sum > target:
                    right -= 1
                else:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                    
    return result

# Example usage:
nums = [1, 0, -1, 0, -2, 2]
target = 0
result = fourSum(nums, target)
print(result)  # Output: [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
    """
    
    # Classify code
    likelihood = calculate_ngram_likelihood(ngram_freqs, code_sample)
    perplexity = calculate_perplexity(ngram_freqs, code_sample)
    classification = classify_code_ml(code_sample, model, vectorizer)
    
    # Check if code is copied from the web
    sources = check_web_copy(code_sample)
    
    # Calculate similarity with corpus
    similarities = calculate_similarity(human_corpus, code_sample)
    
    # Display detailed analysis
    display_analysis(code_sample, likelihood, perplexity, classification, sources, similarities)

if __name__ == "__main__":
    main()