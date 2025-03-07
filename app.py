import os
import sys
from flask import Flask, request, jsonify
from src.content_based import ContentBasedRecommender
from src.collaborative_based import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.deep_learning import DeepLearningRecommender  # Import the new class

# Add src directory to the Python path (if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Initialize Flask app
app = Flask(__name__)

# Load preprocessed data paths
CONTENT_DATA_PATH = "Processed data/all_posts_with_features.csv"
INTERACTION_DATA_PATH = "Processed data/interaction_df.csv"
MODEL_PATH = "models/deep_recommender.h5"  # Path for saved model

# Initialize recommendation systems
content_recommender = ContentBasedRecommender(CONTENT_DATA_PATH)
collaborative_recommender = CollaborativeRecommender(INTERACTION_DATA_PATH)

# Initialize the Hybrid Recommender
hybrid_recommender = HybridRecommender(
    content_model=content_recommender,
    collaborative_model=collaborative_recommender
)

# Initialize the Deep Learning Recommender
# Check if model exists, otherwise create a new one
try:
    deep_recommender = DeepLearningRecommender(
        interaction_data_path=INTERACTION_DATA_PATH,
        content_data_path=CONTENT_DATA_PATH,
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None
    )
    
    # Train the model if it's newly created
    if not os.path.exists(MODEL_PATH):
        print("Training the deep learning recommender model...")
        deep_recommender.train(epochs=10, batch_size=64)
        deep_recommender.save_model(MODEL_PATH)
        
    deep_learning_available = True
except Exception as e:
    print(f"Error initializing deep learning recommender: {str(e)}")
    deep_learning_available = False

@app.route('/feed', methods=['GET'])
def get_recommendations():
    username = request.args.get('username')
    category_id = request.args.get('category_id', type=int)
    mood = request.args.get('mood')
    model_type = request.args.get('model_type', 'hybrid')  # Default to hybrid if not specified
    
    if not username:
        return jsonify({"error": "Missing required parameter: username"}), 400

    try:
        # Convert username to integer (if possible)
        try:
            username = int(username)
        except ValueError:
            return jsonify({"error": "Invalid username format. It must be an integer."}), 400

        print(f"API Username Provided: {username} (Type: {type(username)})")
        
        # Choose recommendation model based on request
        if model_type == 'deep' and deep_learning_available:
            print(f"Using Deep Learning Recommender for user {username}")
            recommendations = deep_recommender.recommend(
                user_id=username, 
                category_id=category_id, 
                mood=mood, 
                top_n=10
            )
        elif model_type == 'content':
            print(f"Using Content-Based Recommender for user {username}")
            recommendations = content_recommender.recommend(username, top_n=10)
        elif model_type == 'collaborative':
            print(f"Using Collaborative Recommender for user {username}")
            recommendations = collaborative_recommender.recommend(username, top_n=10)
        else:
            # Default to hybrid
            print(f"Using Hybrid Recommender for user {username}")
            recommendations = hybrid_recommender.recommend_hybrid(username, top_n=10)
            
        # Handle empty recommendations
        if recommendations.empty:
            print(f"No recommendations available for user {username} with provided filters.")
            return jsonify({"error": "No recommendations available with the given filters"}), 404

        # Check if 'category_id' is in the recommendations DataFrame
        if 'category_id' not in recommendations.columns:
            # Assign a default category_id if it's missing
            print("Warning: 'category_id' column not found. Assigning default category_id.")
            recommendations['category_id'] = 1  # Default category

        # Filter by category_id if provided and not already filtered
        if category_id is not None and model_type != 'deep':
            recommendations = recommendations[recommendations['category_id'] == category_id]

        # Check if 'mood_tags' exists and filter by mood if not already filtered
        if mood and model_type != 'deep':
            if 'mood_tags' in recommendations.columns:
                recommendations = recommendations[recommendations['mood_tags'].str.contains(mood, na=False, case=False)]
            else:
                print(f"Warning: 'mood_tags' column not found in recommendations. Skipping mood filter.")

        # Check for empty recommendations after filtering
        if recommendations.empty:
            print(f"No recommendations available for user {username} after applying filters.")
            return jsonify({"error": "No recommendations available after filtering"}), 404

        # Convert recommendations to JSON
        response = recommendations.to_dict(orient="records")
        return jsonify({"recommendations": response})

    except Exception as e:
        print(f"Error encountered: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to manually trigger model training"""
    if not deep_learning_available:
        return jsonify({"error": "Deep learning recommender is not available"}), 500
        
    try:
        # Get training parameters from request
        data = request.get_json() or {}
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 64)
        
        # Train the model
        print(f"Training deep learning model with {epochs} epochs and batch size {batch_size}")
        deep_recommender.train(epochs=epochs, batch_size=batch_size)
        
        # Save the model
        deep_recommender.save_model(MODEL_PATH)
        
        return jsonify({"success": "Model trained and saved successfully"})
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)