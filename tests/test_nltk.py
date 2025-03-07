import nltk

# Download required NLTK datasets
nltk.download('stopwords')  # For stopwords
nltk.download('punkt')      # For tokenization
nltk.download('wordnet')    # For lemmatization
nltk.download('averaged_perceptron_tagger')  # For POS tagging (optional)

from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

# Sample text
text = "Natural Language Processing is a fascinating field of study!"

# Tokenization
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Stopwords removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Tokens:", filtered_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)