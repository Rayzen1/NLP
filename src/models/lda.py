import os
import numpy as np
import json
import gensim
import nltk
from gensim.models import Phrases
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import glob

DATA_DIRECTORY = 'json_data'
NUM_TOPICS = 10
RANDOM_STATE = 42
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer() 

def preprocess_text(text):
    """Preprocesses a single text document."""
    # Tokenize into words
    words = nltk.word_tokenize(text) 

    # Remove stop words
    words = [word for word in words if word.lower() not in nltk.corpus.stopwords.words('english') and 
             word.lower() not in custom_stop_words] 

    words = [lemmatizer.lemmatize(word) for word in words]
    
    bigram_transformer = Phrases(words)  # Assuming 'words' is your tokenized list 
    words = bigram_transformer[words]

    return ' '.join(words)  

def read_stopwords_from_file(filename):
    """Reads a list of stopwords from a text file, one word per line."""
    with open(filename, 'r' ,encoding='utf-8') as f:
        stopwords = f.readlines()
    # Clean up the words (remove newlines, etc.)
    stopwords = [word.strip() for word in stopwords]
    return stopwords
def read_documents(directory_path):
    """Reads and returns the corpus from the given directory path."""
    corpus = []
    with open(directory_path, 'r') as file:
        data = json.load(file)
        for instance in data: 
            src = ' '.join([' '.join(sentence) for sentence in instance['src']])
            corpus.append(src) 
    return corpus
custom_stop_words = read_stopwords_from_file('C:\Users\Rayzen\Downloads\NLP\src\models\stop_words_english.txt')
def create_lda_model(corpus, num_topics, random_state):
    """Creates and returns an LDA model trained on the given corpus."""
    processed_corpus = [preprocess_text(doc) for doc in corpus]
    vectorizer = CountVectorizer(stop_words=custom_stop_words, min_df=0.1, max_df=0.90)
    doc_term_matrix = vectorizer.fit_transform(processed_corpus)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=random_state, learning_method='online')
    lda_model.fit(doc_term_matrix)
    return lda_model, vectorizer

def get_topic_embedding(lda_model, vectorizer, text):
    """Returns the topic embedding for the given text."""
    x = vectorizer.transform([text])
    topic_distribution = lda_model.transform(x)[0]
    return topic_distribution

for name in glob.glob('JSON/*.json'): 
    print(name)
    corpus = read_documents(name)
    lda_model, vectorizer = create_lda_model(corpus, NUM_TOPICS, RANDOM_STATE)
    topic_embeddings = [get_topic_embedding(lda_model, vectorizer, text) for text in corpus]
    topic_embeddings = [embedding.tolist() for embedding in topic_embeddings]
    # Create a new file name based on the original JSON file name
    base_name = os.path.basename(name)
    new_name = os.path.join('topic', base_name.replace('.json', '.json'))
    with open(new_name, 'w') as f:
        json.dump(topic_embeddings, f)
    print(f"Topic embeddings saved to {new_name}")
