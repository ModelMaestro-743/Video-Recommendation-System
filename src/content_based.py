import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

class ContentBasedRecommender:
    def __init__(self, posts_path):
        self.posts_df = pd.read_csv(posts_path)

        # Inspect the first few rows to verify correct loading
        print(self.posts_df.head())

        # Remove extra spaces from column names (if any)
        self.posts_df.columns = self.posts_df.columns.str.strip()

        # Ensure 'category_id' exists or add a default value column if missing
        if 'category_id' not in self.posts_df.columns:
            print("Warning: 'category_id' column is missing. Adding default category_id.")
            self.posts_df['category_id'] = -1  # Set to a default value (e.g., -1)

        self.posts_df['category_id'] = self.posts_df['category_id'].fillna(-1)  # Handle any missing category_id values
        self._prepare_data()

    def _prepare_data(self):
        if self.posts_df.empty:
            print("Posts DataFrame is empty. Skipping data preparation.")
            self.title_tfidf_matrix = None
            self.category_encoded = None
            self.moods_encoded = None
            self.combined_features = None
            self.similarity_matrix = None
            return

        # Preprocess text for titles
        self.posts_df['processed_title'] = self.posts_df['title'].apply(self._preprocess_text)

        # Vectorize titles using TF-IDF
        self.tfidf = TfidfVectorizer()
        self.title_tfidf_matrix = self.tfidf.fit_transform(self.posts_df['processed_title'])

        # One-hot encode categories
        self.category_encoded = pd.get_dummies(self.posts_df['category_name'])

        # Process moods: handle NaN or non-string values and vectorize
        self.posts_df['moods'] = self.posts_df['moods'].fillna("").astype(str)  # Ensure all values are strings
        self.posts_df['processed_moods'] = self.posts_df['moods'].apply(lambda x: ' '.join(x.split(',')))  # Treat as text
        self.moods_tfidf_matrix = self.tfidf.fit_transform(self.posts_df['processed_moods'])

        # Combine all features (Title + Category + Moods)
        self.combined_features = hstack([self.title_tfidf_matrix, self.category_encoded.values, 0.5 * self.moods_tfidf_matrix])

        # Compute similarity matrix (cosine similarity)
        self.similarity_matrix = cosine_similarity(self.combined_features, self.combined_features)

    def _preprocess_text(self, text):
        # Ensure the text is a string, and handle NaN or None values
        if not isinstance(text, str):
            text = str(text) if text is not None else ""  # Convert to empty string if None

        # Text cleaning and tokenization
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization and Lemmatization
        tokenizer = TreebankWordTokenizer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        tokens = [word for word in tokenizer.tokenize(text) if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(lemmatized)

    def recommend(self, post_id, top_n=10, category_id=None, mood=None):
        if self.similarity_matrix is None:
            print("Similarity matrix not computed. Unable to provide recommendations.")
            return pd.DataFrame()

        if post_id not in self.posts_df['id'].values:
            print(f"Post ID {post_id} not found in posts data.")
            return pd.DataFrame(columns=['post_id', 'score'])

        # Locate the post index
        post_index = self.posts_df[self.posts_df['id'] == post_id].index[0]

        # Apply category and mood filtering before generating recommendations
        filtered_posts = self.posts_df

        if category_id:
            # Check if category_name column exists and filter accordingly
            if 'category_name' in filtered_posts.columns:
                filtered_posts = filtered_posts[filtered_posts['category_name'] == category_id]  # Filter by category_name
            else:
                print("category_name column is not available for filtering.")
                return pd.DataFrame()

        if mood:
            filtered_posts = filtered_posts[filtered_posts['moods'].str.contains(mood, case=False, na=False)]

        if filtered_posts.empty:
            print(f"No posts found matching the category or mood filters.")
            return pd.DataFrame(columns=['post_id', 'score'])

        # Recalculate the similarity matrix for the filtered posts
        filtered_title_tfidf_matrix = self.tfidf.transform(filtered_posts['processed_title'])
        filtered_moods_tfidf_matrix = self.tfidf.transform(filtered_posts['processed_moods'])
        filtered_combined_features = hstack([filtered_title_tfidf_matrix, self.category_encoded.loc[filtered_posts.index].values, 0.5 * filtered_moods_tfidf_matrix])
        filtered_similarity_matrix = cosine_similarity(filtered_combined_features, filtered_combined_features)

        # Calculate similarity scores for the filtered posts
        similarity_scores = list(enumerate(filtered_similarity_matrix[post_index]))

        # Sort and retrieve top N recommendations
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        recommendations = [filtered_posts.iloc[i[0]]['id'] for i in similarity_scores]

        return pd.DataFrame({'post_id': recommendations, 'score': [score for _, score in similarity_scores]})


    def _recommend_cold_start(self, user_mood, top_n):
        # Normalize the user mood to lowercase
        user_mood = user_mood.lower()

        # Split the moods in the dataset into a list of individual moods
        self.posts_df['moods_list'] = self.posts_df['moods'].apply(lambda x: [mood.strip().lower() for mood in x.split(',')] if isinstance(x, str) else [])

        # Now check if the user mood is a substring of any post's moods list
        matching_posts = self.posts_df[self.posts_df['moods_list'].apply(lambda moods: any(user_mood in mood for mood in moods))]

        if matching_posts.empty:
            print(f"No matching posts found for mood: {user_mood}")
            return pd.DataFrame()

        # Create a score column for the cold start recommendations (for now, we set it to a constant value, e.g., 1.0)
        matching_posts['score'] = 1.0  # You can modify this logic to assign different scores based on additional criteria

        # Return the top N recommendations, sorted by score (descending)
        recommendations = matching_posts[['id', 'score']].sort_values(by='score', ascending=False).head(top_n)
        
        # Reset index and return
        return recommendations.reset_index(drop=True)