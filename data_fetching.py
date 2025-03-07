import requests
import pandas as pd
import json
import time
from datetime import datetime
import os

class SocialVerseDataCollector:
    def __init__(self, api_base_url="https://api.socialverseapp.com", headers=None):
        self.api_base_url = api_base_url
        self.headers = headers or {}
        self.resonance_param = "resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        self.output_dir = "socialverse_data"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_data(self, endpoint, params=None):
        """Fetch data from API endpoint with pagination support"""
        url = f"{self.api_base_url}/{endpoint}"
        all_posts = []
        page = 1
        page_size = 1000
        max_page = None
        
        while True:
            query_params = {
                "page": page,
                "page_size": page_size
            }
            
            if params:
                query_params.update(params)
                
            try:
                # Always use headers for all requests
                response = requests.get(url, params=query_params, headers=self.headers)
                response.raise_for_status()
                
                response_data = response.json()
                
                # Debug output to understand response structure
                print(f"Response structure for {endpoint}: {json.dumps(response_data)[:200]}...")
                
                # More flexible data extraction logic
                posts = []
                if "posts" in response_data:
                    posts = response_data["posts"]
                elif "data" in response_data:
                    posts = response_data["data"]
                elif "users" in response_data:
                    posts = response_data["users"]
                elif isinstance(response_data, list):
                    posts = response_data
                else:
                    # Save the full response for debugging
                    debug_filename = f"{self.output_dir}/debug_response_{endpoint.replace('/', '_')}_page{page}.json"
                    with open(debug_filename, 'w') as f:
                        json.dump(response_data, f, indent=4)
                    print(f"Unknown response structure for {endpoint}. Keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'not a dict'}")
                    
                if posts:
                    all_posts.extend(posts)
                
                # Check if we've reached max page
                max_page = response_data.get("max_page_size") if isinstance(response_data, dict) else None
                current_page = response_data.get("page") if isinstance(response_data, dict) else None
                
                # Break if we're on the last page or no more data
                if max_page and current_page and current_page >= max_page:
                    break
                if not posts:
                    break
                    
                page += 1
                time.sleep(0.5)  # Polite delay to avoid rate limiting
                
            except Exception as e:
                print(f"Error fetching data from {url}: {e}")
                break
                
        return all_posts
    
    def test_endpoints(self):
        """Test each endpoint individually to identify issues"""
        endpoints = [
            f"posts/view?{self.resonance_param}",
            f"posts/like?{self.resonance_param}",
            f"posts/inspire?{self.resonance_param}",
            f"posts/rating?{self.resonance_param}",
            "posts/summary/get",
            "users/get_all"
        ]
        
        results = {}
        for endpoint in endpoints:
            print(f"\n----- Testing endpoint: {endpoint} -----")
            result = self.fetch_data(endpoint)
            results[endpoint] = len(result)
            print(f"Got {len(result)} results from {endpoint}")
            
            # Save a sample of the first response
            if result:
                sample_file = f"{self.output_dir}/sample_{endpoint.replace('/', '_').split('?')[0]}.json"
                with open(sample_file, 'w') as f:
                    json.dump(result[0] if len(result) > 0 else {}, f, indent=4)
        
        print("\n----- Endpoint Test Results -----")
        for endpoint, count in results.items():
            print(f"{endpoint}: {count} records")
    
    def collect_all_data(self):
        """Collect data from all endpoints and save to CSV/JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Viewed Posts
        print("Collecting viewed posts...")
        viewed_posts = self.fetch_data(f"posts/view?{self.resonance_param}")
        self._save_data(viewed_posts, f"viewed_posts_{timestamp}")
        
        # 2. Liked Posts
        print("Collecting liked posts...")
        liked_posts = self.fetch_data(f"posts/like?{self.resonance_param}")
        self._save_data(liked_posts, f"liked_posts_{timestamp}")
        
        # 3. Inspired Posts
        print("Collecting inspired posts...")
        inspired_posts = self.fetch_data(f"posts/inspire?{self.resonance_param}")
        self._save_data(inspired_posts, f"inspired_posts_{timestamp}")
        
        # 4. Rated Posts
        print("Collecting rated posts...")
        rated_posts = self.fetch_data(f"posts/rating?{self.resonance_param}")
        self._save_data(rated_posts, f"rated_posts_{timestamp}")
        
        # 5. All Posts
        print("Collecting all posts...")
        all_posts = self.fetch_data("posts/summary/get")
        self._save_data(all_posts, f"all_posts_{timestamp}")
        
        # 6. All Users
        print("Collecting all users...")
        all_users = self.fetch_data("users/get_all")
        self._save_data(all_users, f"all_users_{timestamp}")
        
        # Create a combined dataset for model training
        self._create_combined_training_dataset(timestamp)
        
    def _save_data(self, data, filename_prefix):
        """Save data to CSV and JSON formats"""
        if not data:
            print(f"No data to save for {filename_prefix}")
            return
            
        # Save as JSON
        json_file = f"{self.output_dir}/{filename_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        # Try to save as CSV if data structure allows
        try:
            # Flatten nested JSON structures
            flattened_data = []
            for item in data:
                # Handle case where item might be a string instead of dict
                if isinstance(item, dict):
                    flat_item = self._flatten_json(item)
                    flattened_data.append(flat_item)
                else:
                    print(f"Warning: Non-dict item in dataset: {type(item)}")
            
            if flattened_data:
                df = pd.DataFrame(flattened_data)
                csv_file = f"{self.output_dir}/{filename_prefix}.csv"
                df.to_csv(csv_file, index=False)
                print(f"Saved {len(data)} records to {json_file} and {csv_file}")
            else:
                print(f"No valid data to save as CSV for {filename_prefix}")
        except Exception as e:
            print(f"Could not convert {filename_prefix} to CSV: {e}")
    
    def _flatten_json(self, json_obj, parent_key='', sep='_'):
        """Flatten nested JSON structures for CSV conversion"""
        items = {}
        for k, v in json_obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and len(v) < 10:  # Only flatten small dicts
                items.update(self._flatten_json(v, new_key, sep=sep))
            else:
                # Convert lists to strings to avoid DataFrame conversion issues
                if isinstance(v, list):
                    v = json.dumps(v)
                items[new_key] = v
        return items
    
    def _create_combined_training_dataset(self, timestamp):
        """Combine all data into a format suitable for model training"""
        try:
            # Load saved datasets
            liked_posts = self._load_json(f"{self.output_dir}/liked_posts_{timestamp}.json")
            viewed_posts = self._load_json(f"{self.output_dir}/viewed_posts_{timestamp}.json")
            inspired_posts = self._load_json(f"{self.output_dir}/inspired_posts_{timestamp}.json")
            rated_posts = self._load_json(f"{self.output_dir}/rated_posts_{timestamp}.json")
            all_posts = self._load_json(f"{self.output_dir}/all_posts_{timestamp}.json")
            all_users = self._load_json(f"{self.output_dir}/all_users_{timestamp}.json")
            
            # Create a mapping of post IDs to content details
            post_details = {}
            for post in all_posts:
                if isinstance(post, dict) and "id" in post:
                    post_details[post["id"]] = post
            
            # Create user details mapping
            user_details = {}
            for user in all_users:
                if isinstance(user, dict) and "id" in user:
                    user_details[user["id"]] = user
            
            # Create a combined dataset with user interactions
            training_data = []
            
            # Process each type of interaction
            post_types = {
                "liked": liked_posts,
                "viewed": viewed_posts,
                "inspired": inspired_posts,
                "rated": rated_posts
            }
            
            for interaction_type, posts in post_types.items():
                for post in posts:
                    if not isinstance(post, dict):
                        continue
                    
                    post_id = post.get("id") or post.get("post_id")
                    user_id = post.get("user_id")
                    
                    if not post_id or not user_id:
                        continue
                    
                    # Get post details
                    post_content = post_details.get(post_id, {})
                    user_info = user_details.get(user_id, {})
                    
                    record = {
                        "post_id": post_id,
                        "user_id": user_id,
                        "interaction_type": interaction_type,
                        "interaction_value": 1 if interaction_type != "rated" else post.get("rating_value", 0),
                        "timestamp": post.get("created_at"),
                        # Add post details
                        "category": post_content.get("category"),
                        "content_type": post_content.get("content_type"),
                        "tags": json.dumps(post_content.get("tags", [])),
                        "view_count": post_content.get("view_count", 0),
                        "like_count": post_content.get("like_count", 0),
                        # Add user details
                        "user_age": user_info.get("age"),
                        "user_gender": user_info.get("gender"),
                        "user_interests": json.dumps(user_info.get("interests", []))
                    }
                    training_data.append(record)
            
            # Save the combined training dataset
            if training_data:
                training_df = pd.DataFrame(training_data)
                training_df.to_csv(f"{self.output_dir}/training_dataset_{timestamp}.csv", index=False)
                print(f"Created combined training dataset with {len(training_data)} records")
            else:
                print("No training data could be created - check if your data collection was successful")
            
        except Exception as e:
            print(f"Error creating combined dataset: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_json(self, filepath):
        """Load JSON data from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: File {filepath} does not exist")
                return []
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []


if __name__ == "__main__":
    # Use the Flic-Token header format from the readme
    headers = {
        "Flic-Token": "flic_11d3da28e403d182c36a3530453e290add87d0b4a40ee50f17611f180d47956f",
        "Content-Type": "application/json"
    }
    
    collector = SocialVerseDataCollector(headers=headers)
    
    # First test each endpoint individually to diagnose issues
    print("Testing individual endpoints...")
    collector.test_endpoints()
    
    # Then collect all data
    print("\nCollecting all data...")
    collector.collect_all_data()
    print("Data collection complete.")