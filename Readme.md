# Video Recommendation System

This project implements a recommendation system for the Socialverse platform using a hybrid approach that combines **content-based** and **collaborative** filtering methods. It utilizes user interactions, post features, and mood-based filtering to generate personalized recommendations for users. 


## 🎯 Project Overview

This project implements a video recommendation algorithm that:

- Delivers personalized content recommendations
- Handles cold start problems using mood-based recommendations
- Utilizes deep neural networks for content analysis
- Integrates with external APIs for data collection
- Implements efficient data caching and pagination

## Running the Project

To run the project locally, follow the steps below:

### 1. Install Dependencies

Ensure you have Python 3.x installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the Flask Application

Run the application using the following command:
```bash
python3 app.py
```
This will start the Flask server on `http://127.0.0.1:5000`.

### 3. Access the API

Once the application is running, you can interact with the API by sending a request to the /feed endpoint with the desired parameters.

### Usage

#### 1. Get Recommended Posts with Category and Mood Filters

**Endpoint**: `/feed`  
**Method**: `GET`

##### Request Parameters:
- `username` (required): The ID of the user (integer).
- `category_id` (optional): The ID of the category the user wants to view.
- `mood` (optional): The mood the user is currently in (e.g., `happy`, `passion`, etc.).

##### URL Example:
```
http://127.0.0.1:5000/feed?username=12&category_id=1&mood=passion
```

## 📊 API Endpoints

### 1. Get Recommendations  
**URL:** `/feed`  

**Method:** `GET`  

#### **Parameters:**
- `username` (required): User ID.  
- `category_id` (optional): Filter recommendations by category.  
- `mood` (optional): Filter recommendations by mood.  

#### **Example Request:**
```bash
GET /feed?username=211&category_id=1&mood=passion
```

### **Example Response:**
{
  "recommendations": [
    {
      "category_id": 1,
      "post_id": 366,
      "weighted_score": 0.456
    },
    {
      "category_id": 1,
      "post_id": 17,
      "weighted_score": 0.279
    }
  ]
}

### **License**
This project is licensed under the MIT License. See the LICENSE file for details.

### **Acknowledgments**
Thanks to the developers of Flask, SQLAlchemy, and Alembic for their amazing tools.

Inspiration from Netflix and YouTube recommendation systems.

### **Contact**
For questions or feedback, feel free to reach out:

Email: gaikwadshreya743@gmail.com

Name: Shreya Gaikwad