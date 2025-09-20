# Agriverse - AI-Powered Agricultural Management Platform

Agriverse is a comprehensive AI-powered agricultural management platform connecting farmers, investors, and administrators. The platform leverages advanced machine learning models for crop prediction, yield forecasting, and soil analysis to provide data-driven agricultural insights.

## 🤖 AI/ML Features

### Advanced Machine Learning Models
- **Crop Prediction Model**: Random Forest classifier with 85-90% accuracy
- **Yield Forecasting Model**: Random Forest regressor with comprehensive evaluation metrics
- **Soil Analysis Model**: K-means clustering and health assessment
- **Real-time Predictions**: RESTful API for instant ML predictions
- **Model Management**: Admin interface for training and monitoring models

### ML Capabilities
- **Soil Data Preprocessing**: Advanced feature engineering and normalization
- **Weather Integration**: Temperature, humidity, and rainfall analysis
- **Crop Recommendations**: AI-powered crop selection with confidence scores
- **Yield Predictions**: Accurate yield forecasting with confidence intervals
- **Soil Health Analysis**: Comprehensive soil health scoring and recommendations
- **Batch Processing**: Handle multiple predictions simultaneously

## 🌾 Platform Features

### Farmer Dashboard
- **AI Crop Predictor**: ML-based crop recommendations with confidence scores
- **Yield Forecasting**: Predict crop yields with accuracy metrics
- **Soil Health Analysis**: Comprehensive soil analysis and improvement recommendations
- **Smart Product Sales**: AI-enhanced product recommendations
- **Investment Management**: Seek and manage agricultural investments
- **Performance Analytics**: Data-driven insights and reporting
- **Government Scheme Applications**: Streamlined application process
- **Real-time Weather Integration**: Weather-based decision support

### Investor Dashboard
- **AI-Enhanced Analytics**: ML-powered investment insights
- **Farmer Performance Metrics**: Data-driven farmer evaluation
- **Risk Assessment**: AI-based investment risk analysis
- **Market Predictions**: ML-powered market trend analysis
- **Portfolio Management**: Advanced investment tracking

### Admin Dashboard
- **ML Model Management**: Train, monitor, and deploy ML models
- **Model Performance Metrics**: Real-time model accuracy and performance tracking
- **Cache Management**: Redis-based caching for optimal performance
- **User Analytics**: Comprehensive user behavior analysis
- **System Monitoring**: Real-time system health and performance metrics

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Redis (for caching)
- PostgreSQL/MySQL (for production)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/agriverse.git
   cd agriverse
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   # Application
   FLASK_APP=app
   FLASK_ENV=development
   SECRET_KEY=your-super-secret-key-change-this
   
   # Database
   DATABASE_URL=sqlite:///agriverse.db  # Development
   # DATABASE_URL=postgresql://user:password@localhost/agriverse  # Production
   
   # Redis Cache
   REDIS_URL=redis://localhost:6379/0
   
   # Email Configuration
   MAIL_SERVER=smtp.gmail.com
   MAIL_PORT=587
   MAIL_USE_TLS=True
   MAIL_USERNAME=your-email@gmail.com
   MAIL_PASSWORD=your-app-specific-password
   
   # Stripe Configuration
   STRIPE_SECRET_KEY=your-stripe-secret-key
   STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key
   
   # Oracle Cloud (Optional)
   OCI_CONFIG_PATH=~/.oci/config
   OCI_PROFILE=DEFAULT
   ```

5. **Initialize the database**
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

6. **Train ML Models**
   ```bash
   # Train all models
   curl -X POST http://localhost:5000/api/ml/train-all-models
   
   # Or train individually
   curl -X POST http://localhost:5000/api/ml/train-crop-model
   curl -X POST http://localhost:5000/api/ml/train-yield-model
   ```

7. **Run the application**
   ```bash
      Python run.py
  
   ```

### Production Deployment

1. **Use Production Configuration**
   ```bash
   export FLASK_ENV=production
   export DATABASE_URL=postgresql://user:password@localhost/agriverse
   ```

2. **Deploy with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 run:app
   ```

3. **Docker Deployment**
   ```bash
   docker build -t agriverse .
   docker run -p 8000:8000 agriverse
   ```

## Project Structure

```
agriverse/
├── app/
│   ├── __init__.py
│   ├── models/
│   ├── routes/
│   ├── static/
│   └── templates/
├── migrations/
├── instance/
├── .env
├── .gitignore
├── config.py
├── requirements.txt
└── run.py
```

## Database Schema

The application uses SQLite for development and can be configured to use PlanetScale (MySQL) for production. The main tables include:
- Users (with role-based authentication)
- Products
- Investments
- Soil Reports
- Transactions
- Government Schemes

## Authentication

The platform implements role-based authentication with three user types:
- Farmer
- Investor
- Admin

Each role has specific access permissions and dedicated dashboards.

## 🔌 API Endpoints

### ML API Endpoints

#### Crop Prediction
```bash
# Predict crop recommendations
POST /api/ml/predict-crop
Content-Type: application/json

{
  "ph_level": 6.5,
  "nitrogen_level": 60,
  "phosphorus_level": 50,
  "potassium_level": 55,
  "organic_matter": 45,
  "moisture_content": 65,
  "temperature": 25,
  "humidity": 70,
  "rainfall": 5
}
```

#### Yield Forecasting
```bash
# Predict crop yield
POST /api/ml/predict-yield
Content-Type: application/json

{
  "ph_level": 6.5,
  "nitrogen_level": 60,
  "phosphorus_level": 50,
  "potassium_level": 55,
  "organic_matter": 45,
  "moisture_content": 65,
  "crop_type": "Rice",
  "temperature": 25,
  "humidity": 70,
  "rainfall": 5
}
```

#### Soil Analysis
```bash
# Analyze soil health
POST /api/ml/analyze-soil
Content-Type: application/json

{
  "ph_level": 6.5,
  "nitrogen_level": 60,
  "phosphorus_level": 50,
  "potassium_level": 55,
  "organic_matter": 45,
  "moisture_content": 65
}
```

#### Batch Predictions
```bash
# Process multiple samples
POST /api/ml/batch-predict
Content-Type: application/json

{
  "samples": [
    {
      "ph_level": 6.5,
      "nitrogen_level": 60,
      "phosphorus_level": 50,
      "potassium_level": 55,
      "organic_matter": 45,
      "moisture_content": 65
    }
  ]
}
```

### Model Management
```bash
# Train all models
POST /api/ml/train-all-models

# Get model performance
GET /api/ml/model-performance

# Train specific model
POST /api/ml/train-crop-model
POST /api/ml/train-yield-model
```

## 📊 ML Model Performance

### Crop Prediction Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 85-90%
- **Features**: 15 engineered features
- **Evaluation**: Cross-validation with 5 folds

### Yield Forecasting Model
- **Algorithm**: Random Forest Regressor
- **Metrics**: MAE, RMSE, R², MAPE
- **Features**: 24 engineered features
- **Confidence Intervals**: 95% confidence level

### Soil Analysis Model
- **Algorithm**: K-means Clustering + Rule-based Analysis
- **Clusters**: 3-5 optimal soil types
- **Health Scoring**: 0-100 scale
- **Recommendations**: Automated improvement suggestions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
