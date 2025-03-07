import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import os

class DeepLearningRecommender:
    def __init__(self, interaction_data_path, content_data_path, model_path=None):
        """
        Initialize the Deep Learning Recommender
        
        Args:
            interaction_data_path (str): Path to the interaction data CSV
            content_data_path (str): Path to the content features CSV
            model_path (str, optional): Path to load a pre-trained model
        """
        self.interaction_data_path = interaction_data_path
        self.content_data_path = content_data_path
        self.model_path = model_path
        
        # Load and preprocess data
        self.load_data()
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.model = load_model(model_path)
        else:
            print("Initializing new model")
            self.build_model()
    
    def load_data(self):
        """Load and preprocess the interaction and content data"""
        print("Loading interaction and content data...")
        
        # Load interaction data
        self.interactions_df = pd.read_csv(self.interaction_data_path)
        
        # Load content data
        self.content_df = pd.read_csv(self.content_data_path)
        
        # Encode user IDs and post IDs
        self.user_encoder = LabelEncoder()
        self.post_encoder = LabelEncoder()
        
        self.interactions_df['user_encoded'] = self.user_encoder.fit_transform(self.interactions_df['user_id'])
        self.interactions_df['post_encoded'] = self.post_encoder.fit_transform(self.interactions_df['post_id'])
        
        # Get number of users and posts for embedding dimensions
        self.num_users = len(self.user_encoder.classes_)
        self.num_posts = len(self.post_encoder.classes_)
        
        # Create a mapping from post_id to content features
        self.post_id_to_features = {}
        
        # Extract text and categorical features from content data
        if 'text_embedding' in self.content_df.columns:
            # If text embeddings are already available
            self.text_feature_dim = len(eval(self.content_df['text_embedding'].iloc[0])) if not self.content_df.empty else 0
        else:
            # Use simpler features if text embeddings are not available
            self.text_feature_dim = 0
        
        # Extract numerical features from content data (e.g., post length, views, etc.)
        numerical_features = ['category_id']  # Add any other numerical features
        
        for _, row in self.content_df.iterrows():
            post_id = row['post_id']
            feature_dict = {
                'category_id': row['category_id'] if 'category_id' in row else 0,
            }
            
            # Add text embeddings if available
            if 'text_embedding' in row and self.text_feature_dim > 0:
                feature_dict['text_embedding'] = eval(row['text_embedding'])
            
            self.post_id_to_features[post_id] = feature_dict
        
        print(f"Loaded {self.num_users} users and {self.num_posts} posts.")
    
    def build_model(self):
        """Build the neural network model for the recommendation system"""
        # Define embedding dimensions
        user_embedding_dim = 32
        post_embedding_dim = 32
        
        # User input and embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(input_dim=self.num_users, output_dim=user_embedding_dim, 
                                  name='user_embedding')(user_input)
        user_embedding = Flatten()(user_embedding)
        
        # Post input and embedding
        post_input = Input(shape=(1,), name='post_input')
        post_embedding = Embedding(input_dim=self.num_posts, output_dim=post_embedding_dim, 
                                  name='post_embedding')(post_input)
        post_embedding = Flatten()(post_embedding)
        
        # Category input (if available)
        category_input = Input(shape=(1,), name='category_input')
        category_embedding = Embedding(input_dim=20, output_dim=8, name='category_embedding')(category_input)
        category_embedding = Flatten()(category_embedding)
        
        # Concatenate all embeddings
        concat = Concatenate()([user_embedding, post_embedding, category_embedding])
        
        # Hidden layers
        x = Dense(128, activation='relu')(concat)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # Define the model
        self.model = Model(
            inputs=[user_input, post_input, category_input],
            outputs=output
        )
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Deep learning recommendation model built successfully.")
        self.model.summary()
    
    def prepare_training_data(self):
        """Prepare the data for training the model"""
        # Get positive interactions (viewed, liked, etc.)
        positive_interactions = self.interactions_df[self.interactions_df['interaction_type'] > 0]
        
        # Generate negative samples (posts not interacted with)
        all_user_post_pairs = set(zip(self.interactions_df['user_id'], self.interactions_df['post_id']))
        all_users = set(self.interactions_df['user_id'])
        all_posts = set(self.interactions_df['post_id'])
        
        negative_samples = []
        for user in all_users:
            # Sample some random posts that the user hasn't interacted with
            posts_interacted = set(self.interactions_df[self.interactions_df['user_id'] == user]['post_id'])
            posts_not_interacted = all_posts - posts_interacted
            
            # Limit the number of negative samples per user to balance the dataset
            n_samples = min(len(posts_not_interacted), len(posts_interacted) * 3)
            if n_samples > 0:
                sampled_posts = np.random.choice(list(posts_not_interacted), size=n_samples, replace=False)
                for post in sampled_posts:
                    negative_samples.append((user, post, 0))  # 0 indicates no interaction
        
        # Create a DataFrame for negative samples
        negative_df = pd.DataFrame(negative_samples, columns=['user_id', 'post_id', 'interaction_type'])
        
        # Encode user and post IDs for negative samples
        negative_df['user_encoded'] = self.user_encoder.transform(negative_df['user_id'])
        negative_df['post_encoded'] = self.post_encoder.transform(negative_df['post_id'])
        
        # Combine positive and negative interactions
        training_df = pd.concat([
            positive_interactions[['user_encoded', 'post_encoded', 'interaction_type']],
            negative_df[['user_encoded', 'post_encoded', 'interaction_type']]
        ])
        
        # Add category information
        training_df['category_id'] = training_df.apply(
            lambda row: self._get_category_for_post(row['post_id']), axis=1
        )
        
        # Convert target to binary (interacted or not)
        training_df['target'] = (training_df['interaction_type'] > 0).astype(int)
        
        # Prepare input arrays
        user_array = training_df['user_encoded'].values
        post_array = training_df['post_encoded'].values
        category_array = training_df['category_id'].values
        target_array = training_df['target'].values
        
        return [user_array, post_array, category_array], target_array
    
    def _get_category_for_post(self, post_id):
        """Get the category ID for a given post"""
        if post_id in self.post_id_to_features:
            return self.post_id_to_features[post_id]['category_id']
        return 0  # Default category if not found
    
    def train(self, epochs=20, batch_size=64, validation_split=0.2):
        """Train the deep learning model"""
        print("Preparing training data...")
        X, y = self.prepare_training_data()
        
        print("Training the model...")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("Model training completed.")
        return history
    
    def save_model(self, model_path='models/deep_recommender.h5'):
        """Save the trained model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def predict_user_preferences(self, user_id, top_n=10):
        """
        Predict the top N posts that a user might be interested in
        
        Args:
            user_id: The user ID to make predictions for
            top_n: Number of top recommendations to return
            
        Returns:
            DataFrame: Top N recommended posts with scores
        """
        # Check if user exists
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except:
            print(f"User {user_id} not found in training data")
            return pd.DataFrame()
        
        # Prepare data for all possible posts
        all_posts = self.post_encoder.classes_
        all_posts_encoded = self.post_encoder.transform(all_posts)
        
        # Get categories for all posts
        categories = [self._get_category_for_post(post_id) for post_id in all_posts]
        
        # Create input arrays for prediction
        user_array = np.full(len(all_posts), user_encoded)
        post_array = all_posts_encoded
        category_array = np.array(categories)
        
        # Make predictions
        predictions = self.model.predict([user_array, post_array, category_array])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'post_id': all_posts,
            'score': predictions.flatten()
        })
        
        # Sort by prediction score (descending)
        results = results.sort_values('score', ascending=False)
        
        # Get top N recommendations
        top_recommendations = results.head(top_n)
        
        # Merge with content data to include additional info
        recommendations = pd.merge(
            top_recommendations,
            self.content_df,
            on='post_id',
            how='left'
        )
        
        return recommendations
    
    def recommend(self, user_id, category_id=None, mood=None, top_n=10):
        """
        Get recommendations for a user with optional filters
        
        Args:
            user_id: The user ID to get recommendations for
            category_id: Optional category filter
            mood: Optional mood filter
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame: Filtered recommendations
        """
        # Get base recommendations
        recommendations = self.predict_user_preferences(user_id, top_n=top_n*2)  # Get more than needed to allow for filtering
        
        if recommendations.empty:
            return recommendations
        
        # Filter by category_id if provided
        if category_id is not None:
            recommendations = recommendations[recommendations['category_id'] == category_id]
        
        # Filter by mood if provided and if mood_tags column exists
        if mood and 'mood_tags' in recommendations.columns:
            recommendations = recommendations[recommendations['mood_tags'].str.contains(mood, na=False, case=False)]
        
        # Return top N after filtering
        return recommendations.head(top_n)