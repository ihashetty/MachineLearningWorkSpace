"""
FastAPI Application for Customer Clustering and Personalized Offers
This application loads a pre-trained KMeans model and provides REST API endpoints
to predict customer segments and generate personalized offers.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List

app = FastAPI(
    title="Customer Clustering API",
    description="Predict customer segments and generate personalized offers",
    version="1.0.0"
)

try:
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    print("✓ KMeans model loaded successfully")
except FileNotFoundError:
    print("✗ Error: kmeans_model.pkl not found")
    kmeans_model = None

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully")
except FileNotFoundError:
    print("✗ Error: scaler.pkl not found")
    scaler = None

try:
    with open('cluster_segments.pkl', 'rb') as f:
        cluster_segments = pickle.load(f)
    print("✓ Cluster segments mapping loaded successfully")
except FileNotFoundError:
    print("✗ Error: cluster_segments.pkl not found")
    cluster_segments = None

PERSONALIZED_OFFERS = {
    0: {  
        "discount": "15% premium loyalty discount",
        "offers": [
            "Exclusive VIP access to new products",
            "Priority customer support (24/7)",
            "Free shipping on all orders",
            "Double loyalty points on purchases",
            "Exclusive member-only sales events"
        ],
        "recommendation": "Maintain premium tier status, provide exclusive benefits"
    },
    1: {  
        "discount": "10% loyalty discount",
        "offers": [
            "Seasonal promotions and discounts",
            "Buy more save more offers",
            "Referral bonus program",
            "Birthday special discounts",
            "Newsletter exclusive deals"
        ],
        "recommendation": "Encourage regular purchases, provide value-based promotions"
    },
    2: {  
        "discount": "20% first purchase discount or bundle offers",
        "offers": [
            "Flash sales and limited-time offers",
            "Bundle deals (3 items at special price)",
            "Seasonal clearance sales",
            "Free shipping on minimum purchase",
            "Try-before-you-buy incentives"
        ],
        "recommendation": "Focus on aggressive discounts, volume deals, and low-commitment offerings"
    }
}


class CustomerInput(BaseModel):
    # Demographic Features
    age: float
    gender: str  # "M" or "F"
    city: str    # City name (e.g., "Bangalore", "Mumbai", "Delhi", "Pune")
    
    # Financial Features
    annual_income: float
    total_spent: float
    monthly_purchases: float
    avg_order_value: float
    
    # Behavioral Features
    app_time_minutes: float
    discount_usage: str  # "Low", "Medium", "High"
    preferred_shopping_time: str  # "Day" or "Night"

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "M",
                "city": "Mumbai",
                "annual_income": 1200000,
                "total_spent": 850000,
                "monthly_purchases": 18,
                "avg_order_value": 12000,
                "app_time_minutes": 120,
                "discount_usage": "Low",
                "preferred_shopping_time": "Night"
            }
        }

class CustomerSegmentResponse(BaseModel):
    customer_id: str
    cluster: int
    segment_name: str
    confidence_score: float
    personalized_offers: Dict
    business_insights: Dict

@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint"""
    return {
        "message": "Customer Clustering API is running",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/predict-segment", response_model=CustomerSegmentResponse, tags=["Prediction"])
def predict_customer_segment(customer: CustomerInput):
    """
    Predict customer segment based on comprehensive customer characteristics.
    
    Features Include:
    - Demographic: Age, Gender, City
    - Financial: Annual Income, Total Spent, Monthly Purchases, Average Order Value
    - Behavioral: App Time, Discount Usage, Shopping Time Preference
    
    Args:
        customer: CustomerInput object with all customer details
    
    Returns:
        CustomerSegmentResponse with cluster prediction and personalized offers
    """
    
    if kmeans_model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Models not loaded. Ensure kmeans_model.pkl and scaler.pkl exist."
        )
    
    try:
        # Encode categorical features
        gender_encoded = 1 if customer.gender.upper() == 'M' else 0
        discount_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        discount_encoded = discount_mapping.get(customer.discount_usage, 0)
        shopping_time_encoded = 1 if customer.preferred_shopping_time.lower() == 'night' else 0
        
        # City one-hot encoding (for known cities from training data)
        cities = ['Bangalore', 'Mumbai', 'Delhi', 'Pune']
        city_features = [1 if customer.city == city else 0 for city in cities[1:]]  # Drop first for one-hot
        
        # Build feature vector in same order as training
        input_data = np.array([[
            customer.age,
            gender_encoded,
            discount_encoded,
            shopping_time_encoded,
            customer.annual_income,
            customer.total_spent,
            customer.monthly_purchases,
            customer.avg_order_value,
            customer.app_time_minutes
        ] + city_features])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict cluster
        cluster = kmeans_model.predict(input_scaled)[0]
        
        # Calculate confidence score
        distances = np.linalg.norm(input_scaled - kmeans_model.cluster_centers_, axis=1)
        min_distance = distances.min()
        max_distance = distances.max()
        confidence = ((max_distance - min_distance) / max_distance) * 100 if max_distance > 0 else 0
        
        # Get segment name
        segment_name = cluster_segments.get(cluster, f"Cluster {cluster}")
        
        # Get personalized offers
        offers = PERSONALIZED_OFFERS.get(cluster, {})
        
        # Generate business insights
        insights = {
            "risk_score": round(confidence, 2),
            "recommendation_priority": "HIGH" if cluster == 0 else ("MEDIUM" if cluster == 1 else "LOW"),
            "targeting_strategy": offers.get("recommendation", ""),
            "estimated_ltv_category": "Premium" if cluster == 0 else ("Standard" if cluster == 1 else "Budget"),
            "features_analyzed": {
                "demographic": ["Age", "Gender", "City"],
                "financial": ["Annual Income", "Total Spent", "Monthly Purchases", "Avg Order Value"],
                "behavioral": ["App Usage Time", "Discount Sensitivity", "Shopping Time Preference"]
            }
        }
        
        return CustomerSegmentResponse(
            customer_id=f"PRED_{np.random.randint(10000, 99999)}",
            cluster=cluster,
            segment_name=segment_name,
            confidence_score=round(confidence, 2),
            personalized_offers={
                "discount": offers.get("discount", ""),
                "offers": offers.get("offers", []),
                "recommendation": offers.get("recommendation", "")
            },
            business_insights=insights
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/batch-predict", tags=["Prediction"])
def batch_predict_segments(customers: List[CustomerInput]):
    """
    Predict segments for multiple customers at once with all encoded features.
    
    Args:
        customers: List of CustomerInput objects with comprehensive customer data
    
    Returns:
        List of prediction results with cluster assignments and offers
    """
    
    if kmeans_model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Models not loaded."
        )
    
    results = []
    for customer in customers:
        try:
            # Encode categorical features
            gender_encoded = 1 if customer.gender.upper() == 'M' else 0
            discount_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            discount_encoded = discount_mapping.get(customer.discount_usage, 0)
            shopping_time_encoded = 1 if customer.preferred_shopping_time.lower() == 'night' else 0
            
            # City one-hot encoding
            cities = ['Bangalore', 'Mumbai', 'Delhi', 'Pune']
            city_features = [1 if customer.city == city else 0 for city in cities[1:]]
            
            # Build feature vector
            input_data = np.array([[
                customer.age,
                gender_encoded,
                discount_encoded,
                shopping_time_encoded,
                customer.annual_income,
                customer.total_spent,
                customer.monthly_purchases,
                customer.avg_order_value,
                customer.app_time_minutes
            ] + city_features])
            
            input_scaled = scaler.transform(input_data)
            cluster = kmeans_model.predict(input_scaled)[0]
            segment_name = cluster_segments.get(cluster, f"Cluster {cluster}")
            offers = PERSONALIZED_OFFERS.get(cluster, {})
            
            distances = np.linalg.norm(input_scaled - kmeans_model.cluster_centers_, axis=1)
            confidence = ((distances.max() - distances.min()) / distances.max() * 100) if distances.max() > 0 else 0
            
            results.append({
                "cluster": cluster,
                "segment_name": segment_name,
                "confidence_score": round(confidence, 2),
                "discount": offers.get("discount", ""),
                "top_offers": offers.get("offers", [])[:3],
                "status": "success"
            })
        except Exception as e:
            results.append({
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "total_processed": len(customers),
        "successful": len([r for r in results if "error" not in r]),
        "results": results
    }

@app.get("/cluster-info", tags=["Information"])
def get_cluster_information():
    """
    Get information about all defined customer segments.
    
    Returns:
        Dictionary with cluster details and offers
    """
    
    cluster_info = {}
    for cluster_id, segment_name in cluster_segments.items():
        cluster_info[f"Cluster {cluster_id}"] = {
            "segment_name": segment_name,
            "discount": PERSONALIZED_OFFERS[cluster_id].get("discount", ""),
            "offers": PERSONALIZED_OFFERS[cluster_id].get("offers", []),
            "targeting_strategy": PERSONALIZED_OFFERS[cluster_id].get("recommendation", "")
        }
    
    return cluster_info

@app.get("/health", tags=["Health Check"])
def health_check():
    """
    Detailed health check endpoint
    """
    return {
        "status": "healthy",
        "models_loaded": {
            "kmeans": kmeans_model is not None,
            "scaler": scaler is not None,
            "cluster_segments": cluster_segments is not None
        },
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("Starting Customer Clustering FastAPI Server")
    print("="*80)
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative API Documentation: http://localhost:8000/redoc")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
