import pandas as pd

class HybridRecommender:
    def __init__(self, content_model, collaborative_model, weight_content=0.3, weight_collaborative=0.7):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.weight_content = weight_content
        self.weight_collaborative = weight_collaborative

    def recommend_hybrid(self, user_id, category_id=None, top_n=10):
        try:
            # Get recommendations from both models
            content_recommendations = self.content_model.recommend(user_id, top_n=top_n)
            collaborative_recommendations = self.collaborative_model.recommend(user_id, top_n=top_n)

            # Handle content-based recommendations
            content_df = content_recommendations if not content_recommendations.empty else pd.DataFrame(columns=["post_id", "score", "category_id"])

            # Handle collaborative recommendations
            if isinstance(collaborative_recommendations, list):
                collaborative_df = pd.DataFrame(
                    collaborative_recommendations,
                    columns=["post_id", "score"] if isinstance(collaborative_recommendations[0], tuple) else ["post_id"]
                )
                if "score" not in collaborative_df.columns:
                    collaborative_df["score"] = 1.0  # Assign default score
                collaborative_df["category_id"] = 1  # Add default category_id if missing
            else:
                collaborative_df = collaborative_recommendations if not collaborative_recommendations.empty else pd.DataFrame(columns=["post_id", "score", "category_id"])

            # Log columns of DataFrames
            print(f"Content DataFrame Columns: {content_df.columns.tolist()}")
            print(f"Collaborative DataFrame Columns: {collaborative_df.columns.tolist()}")

            # Handle category_id filtering
            if category_id:
                if 'category_id' in content_df.columns:
                    content_df = content_df[content_df['category_id'] == category_id]
                else:
                    print("Warning: 'category_id' not found in content recommendations. Skipping filtering.")

                if 'category_id' in collaborative_df.columns:
                    collaborative_df = collaborative_df[collaborative_df['category_id'] == category_id]
                else:
                    print("Warning: 'category_id' not found in collaborative recommendations. Skipping filtering.")

            # Apply weights to both content and collaborative recommendations
            if not content_df.empty:
                content_df["weight"] = self.weight_content
                content_df["weighted_score"] = content_df["score"] * content_df["weight"]

            if not collaborative_df.empty:
                collaborative_df["weight"] = self.weight_collaborative
                collaborative_df["weighted_score"] = collaborative_df["score"] * collaborative_df["weight"]

            # Combine DataFrames safely
            combined_df = pd.concat([content_df.dropna(axis=1, how='all'), collaborative_df.dropna(axis=1, how='all')], ignore_index=True)

            # Aggregate and sort recommendations by weighted score
            final_recommendations = (
                combined_df.groupby("post_id", as_index=False)["weighted_score"]
                .sum()
                .sort_values(by="weighted_score", ascending=False)
                .head(top_n)
            )

            print(f"Final Hybrid Recommendations:\n{final_recommendations}")
            return final_recommendations

        except Exception as e:
            print(f"Error fetching recommendations: {e}")
            raise