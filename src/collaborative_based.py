import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self, interactions_path):
        self.interactions_df = pd.read_csv(interactions_path)
        # print(f"Interactions DataFrame Loaded: {self.interactions_df.shape} rows, columns: {self.interactions_df.columns.tolist()}")
        # print(self.interactions_df.head())  # Display the first few rows for validation
        self._prepare_data()

    def _prepare_data(self):
        # Validate data structure
        if 'user_id' not in self.interactions_df.columns or 'post_id' not in self.interactions_df.columns:
            raise ValueError("Interaction data must contain 'user_id' and 'post_id' columns.")

        # Handle missing or invalid data
        self.interactions_df = self.interactions_df.dropna(subset=['user_id', 'post_id'])
        self.interactions_df['user_id'] = self.interactions_df['user_id'].astype(int)
        self.interactions_df['post_id'] = self.interactions_df['post_id'].astype(int)

        # Create user-post interaction matrix
        self.user_post_matrix = self.interactions_df.pivot_table(
            index='user_id', columns='post_id', values='interaction_type',
            aggfunc=lambda x: 1 if len(x) > 0 else 0
        ).fillna(0)
        # print(f"User-Post Matrix Shape: {self.user_post_matrix.shape}")
        # print("User-Post Interaction Matrix (First Few Rows):")
        # print(self.user_post_matrix.head())

        # Compute item-item similarity
        self.item_similarity_matrix = cosine_similarity(self.user_post_matrix.T)
        # print(f"Item-Item Similarity Matrix Shape: {self.item_similarity_matrix.shape}")

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_post_matrix.index:
            return pd.DataFrame(columns=["post_id", "score"])

        user_interactions = self.user_post_matrix.loc[user_id]
        interacted_posts = user_interactions[user_interactions > 0].index.tolist()

        if not interacted_posts:
            return pd.DataFrame(columns=["post_id", "score"])

        # Compute scores for all posts
        scores = {}
        for post_id in interacted_posts:
            post_idx = self.user_post_matrix.columns.get_loc(post_id)
            similar_posts = self.item_similarity_matrix[post_idx]
            for idx, score in enumerate(similar_posts):
                if self.user_post_matrix.columns[idx] not in interacted_posts:
                    scores[self.user_post_matrix.columns[idx]] = scores.get(self.user_post_matrix.columns[idx], 0) + score

        # Sort scores and retrieve top recommendations
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations_df = pd.DataFrame(recommendations, columns=["post_id", "score"])
        return recommendations_df