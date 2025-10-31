"""
Wildlife Conservation Backend with AI Techniques
Features: 
- Species Classification using KNN
- Population Trend Prediction using Linear Regression
- Conservation Priority Scoring using Random Forest
- Clustering animals by characteristics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import json

app = Flask(__name__)
CORS(app)

# Global variables
dataset = None
models = {}
encoders = {}
scaler = StandardScaler()

def load_dataset():
    """Load dataset from CSV file"""
    global dataset
    try:
        dataset = pd.read_csv('wildlife_dataset.csv')
        print(f"Dataset loaded: {len(dataset)} records")
        return True
    except FileNotFoundError:
        print("Dataset not found. Please generate it first.")
        return False

def train_models():
    """Train AI models on the dataset"""
    global models, encoders, scaler
    
    if dataset is None or len(dataset) == 0:
        return False
    
    # Prepare data
    df = dataset.copy()
    
    # 1. CONSERVATION STATUS CLASSIFIER (KNN)
    # Encode categorical variables
    le_habitat = LabelEncoder()
    le_status = LabelEncoder()
    
    df['habitat_encoded'] = le_habitat.fit_transform(df['habitat'])
    df['status_encoded'] = le_status.fit_transform(df['status'])
    
    encoders['habitat'] = le_habitat
    encoders['status'] = le_status
    
    # Features for classification
    features_classify = ['population', 'lifespan', 'weight', 'height', 
                        'habitat_encoded', 'health_index']
    X_classify = df[features_classify]
    y_classify = df['status_encoded']
    
    # Scale features
    X_classify_scaled = scaler.fit_transform(X_classify)
    
    # Train KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_classify_scaled, y_classify)
    models['status_classifier'] = knn
    
    # 2. CONSERVATION PRIORITY PREDICTOR (Random Forest)
    X_priority = df[['population', 'lifespan', 'health_index', 
                     'habitat_encoded', 'status_encoded']]
    y_priority = df['conservation_score']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Convert to classification (high/medium/low priority)
    y_priority_class = pd.cut(y_priority, bins=[0, 40, 70, 100], 
                              labels=[0, 1, 2])
    rf.fit(X_priority, y_priority_class)
    models['priority_predictor'] = rf
    
    # 3. POPULATION TREND PREDICTOR (Linear Regression)
    lr = LinearRegression()
    X_trend = df[['health_index', 'conservation_score', 'lifespan']]
    y_trend = df['population']
    lr.fit(X_trend, y_trend)
    models['population_predictor'] = lr
    
    # 4. SPECIES CLUSTERING (K-Means)
    kmeans = KMeans(n_clusters=5, random_state=42)
    X_cluster = df[['population', 'lifespan', 'weight', 'height']]
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    df['cluster'] = kmeans.fit_predict(X_cluster_scaled)
    models['species_clusterer'] = kmeans
    
    print("All AI models trained successfully!")
    return True

@app.route('/api/animals', methods=['GET'])
def get_animals():
    """Get all animals or search by name"""
    search = request.args.get('search', '')
    
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    df = dataset.copy()
    if search:
        df = df[df['name'].str.contains(search, case=False) | 
                df['scientific_name'].str.contains(search, case=False)]
    
    return jsonify(df.to_dict('records'))

@app.route('/api/animal/<int:animal_id>', methods=['GET'])
def get_animal(animal_id):
    """Get specific animal details"""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    animal = dataset[dataset['id'] == animal_id]
    if len(animal) == 0:
        return jsonify({'error': 'Animal not found'}), 404
    
    return jsonify(animal.to_dict('records')[0])

@app.route('/api/predict_status', methods=['POST'])
def predict_status():
    """Predict conservation status using KNN"""
    data = request.json
    
    try:
        # Prepare input
        habitat_encoded = encoders['habitat'].transform([data['habitat']])[0]
        
        features = np.array([[
            data['population'],
            data['lifespan'],
            data['weight'],
            data['height'],
            habitat_encoded,
            data['health_index']
        ]])
        
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = models['status_classifier'].predict(features_scaled)[0]
        status = encoders['status'].inverse_transform([prediction])[0]
        
        # Get probability
        probabilities = models['status_classifier'].predict_proba(features_scaled)[0]
        confidence = max(probabilities) * 100
        
        return jsonify({
            'predicted_status': status,
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_priority', methods=['POST'])
def predict_priority():
    """Predict conservation priority using Random Forest"""
    data = request.json
    
    try:
        habitat_encoded = encoders['habitat'].transform([data['habitat']])[0]
        status_encoded = encoders['status'].transform([data['status']])[0]
        
        features = np.array([[
            data['population'],
            data['lifespan'],
            data['health_index'],
            habitat_encoded,
            status_encoded
        ]])
        
        prediction = models['priority_predictor'].predict(features)[0]
        priorities = ['Low Priority', 'Medium Priority', 'High Priority']
        
        return jsonify({
            'priority': priorities[prediction],
            'priority_level': int(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_population', methods=['POST'])
def predict_population():
    """Predict future population using Linear Regression"""
    data = request.json
    
    try:
        features = np.array([[
            data['health_index'],
            data['conservation_score'],
            data['lifespan']
        ]])
        
        prediction = models['population_predictor'].predict(features)[0]
        
        return jsonify({
            'predicted_population': int(max(0, prediction)),
            'trend': 'increasing' if prediction > data.get('current_population', 0) else 'decreasing'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/cluster_analysis', methods=['GET'])
def cluster_analysis():
    """Get cluster analysis of species"""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    try:
        # Re-run clustering if not in dataset
        df = dataset.copy()
        X_cluster = df[['population', 'lifespan', 'weight', 'height']]
        X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
        clusters = models['species_clusterer'].predict(X_cluster_scaled)
        
        df['cluster'] = clusters
        
        # Get cluster statistics
        cluster_stats = []
        for i in range(5):
            cluster_data = df[df['cluster'] == i]
            cluster_stats.append({
                'cluster_id': i,
                'count': len(cluster_data),
                'avg_population': int(cluster_data['population'].mean()),
                'avg_lifespan': round(cluster_data['lifespan'].mean(), 1),
                'common_status': cluster_data['status'].mode()[0] if len(cluster_data) > 0 else 'Unknown'
            })
        
        return jsonify({'clusters': cluster_stats})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall dataset statistics"""
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    stats = {
        'total_species': len(dataset),
        'endangered_count': len(dataset[dataset['status'].isin(['Endangered', 'Critically Endangered'])]),
        'avg_population': int(dataset['population'].mean()),
        'avg_health_index': round(dataset['health_index'].mean(), 2),
        'status_distribution': dataset['status'].value_counts().to_dict(),
        'habitat_distribution': dataset['habitat'].value_counts().to_dict()
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    print("Loading dataset...")
    if load_dataset():
        print("Training AI models...")
        train_models()
        print("Starting Flask server...")
        app.run(debug=True, port=5000)
    else:
        print("Error: Cannot start server without dataset.")
        print("Please run the dataset generator first!")
