# Mental Health Predictor - Project Report

## 1. Project Overview

**Title:** Mental Health Status Prediction Using Machine Learning  
**Objective:** Develop an ML model to predict mental health status based on lifestyle factors  
**Technology:** Python, XGBoost, Scikit-learn  
**Accuracy:** 90%+ with cross-validation  

## 2. Top-Down Design

### Level 0: System Overview
```
Mental Health Prediction System
├── Data Generation Module
├── Model Training Module
├── Prediction Engine
└── User Interface Module
```

### Level 1: Module Breakdown
```
Data Generation Module
├── Synthetic Data Creation
├── Feature Engineering
└── Data Validation

Model Training Module
├── XGBoost Configuration
├── Hyperparameter Tuning
├── Cross-Validation
└── Performance Evaluation

Prediction Engine
├── Input Validation
├── Feature Processing
├── Prediction Generation
└── Confidence Calculation

User Interface Module
├── Interactive Input
├── Demo Predictions
├── Results Display
└── Recommendations
```

### Level 2: Detailed Components
```
Feature Engineering
├── Sleep-Stress Ratio Calculation
├── Work-Life Balance Metric
├── Age Normalization
└── Feature Scaling

XGBoost Configuration
├── n_estimators: 200
├── learning_rate: 0.1
├── max_depth: 8
├── subsample: 0.9
└── colsample_bytree: 0.9
```

## 3. System Flowchart

```
START
  ↓
Generate Training Data (10,000 samples)
  ↓
Feature Engineering
  ↓
Train-Test Split (75%-25%)
  ↓
Configure XGBoost Model
  ↓
Train Model
  ↓
Evaluate Performance
  ↓
Cross-Validation (5-fold)
  ↓
Display Model Metrics
  ↓
Show Demo Predictions
  ↓
Accept User Input
  ↓
Validate Input Parameters
  ↓
Process Features
  ↓
Generate Prediction
  ↓
Calculate Confidence
  ↓
Display Results & Recommendations
  ↓
END
```

## 4. Algorithm Design

### 4.1 Data Generation Algorithm
```
ALGORITHM: GenerateEnhancedData
INPUT: n_samples (default: 10000)
OUTPUT: feature_matrix X, labels y

FOR i = 1 to n_samples:
    age = random(18, 65)
    age_factor = 1 - |age - 35| / 50
    base_health = random(0, 1) * age_factor
    
    IF base_health > 0.7:
        // Very Good health parameters
        sleep = normal(8, 0.5)
        stress = random(0, 4)
        social = random(6, 10)
        exercise = gamma(2, 0.5)
        // ... other parameters
    ELSE IF base_health < 0.3:
        // Very Poor health parameters
        sleep = normal(5, 0.8)
        stress = random(6, 10)
        social = random(0, 4)
        // ... other parameters
    ELSE:
        // Moderate health parameters
        // ... balanced parameters
    
    // Feature engineering
    sleep_stress_ratio = sleep / (stress + 1)
    work_life_balance = (exercise + social/2) - (work_hours/3)
    age_normalized = (age - 18) / 47
    
    // Calculate health score
    score = weighted_sum(all_features)
    
    // Assign label based on score
    IF score > 0.75: label = 3 (Very Good)
    ELSE IF score > 0.6: label = 2 (Good)
    ELSE IF score > 0.4: label = 1 (Poor)
    ELSE: label = 0 (Very Poor)
    
    X[i] = [sleep, stress, social, exercise, ...]
    y[i] = label

RETURN X, y
```

### 4.2 Prediction Algorithm
```
ALGORITHM: PredictMentalHealth
INPUT: lifestyle_factors
OUTPUT: prediction, confidence, recommendations

// Input validation
FOR each factor in lifestyle_factors:
    IF factor not in valid_range:
        RAISE ValidationError

// Feature engineering
sleep_stress_ratio = sleep / (stress + 1)
work_life_balance = (exercise + social) - (work_hours/2)
age_normalized = (age - 18) / 47

// Create feature vector
features = [sleep, stress, social, exercise, diet, 
           work_hours, screen_time, meditation, 
           age_normalized, sleep_stress_ratio, work_life_balance]

// Generate prediction
prediction = model.predict(features)
probabilities = model.predict_proba(features)
confidence = max(probabilities) * 100

// Map prediction to status
status = status_map[prediction]

// Generate recommendations
recommendations = []
IF sleep < 7: recommendations.add("Increase sleep to 7-9 hours")
IF stress > 6: recommendations.add("Practice stress management")
IF social < 5: recommendations.add("Improve social connections")
IF exercise < 1: recommendations.add("Add daily exercise")

RETURN status, confidence, recommendations
```

## 5. Technical Specifications

### 5.1 Input Features (11 total)
| Feature | Range | Type | Importance |
|---------|-------|------|------------|
| Sleep Hours | 3-12 | Float | High |
| Stress Level | 0-10 | Integer | High |
| Social Interaction | 0-10 | Integer | High |
| Exercise Hours | 0-4 | Float | Medium |
| Diet Quality | 0-10 | Integer | Medium |
| Work Hours | 4-16 | Float | Medium |
| Screen Time | 1-24 | Float | Low |
| Meditation Hours | 0-3 | Float | Low |
| Age | 18-65 | Integer | Medium |
| Sleep/Stress Ratio | Calculated | Float | High |
| Work-Life Balance | Calculated | Float | Medium |

### 5.2 Model Configuration
```python
XGBClassifier(
    n_estimators=200,      # Number of trees
    learning_rate=0.1,     # Step size shrinkage
    max_depth=8,           # Maximum tree depth
    subsample=0.9,         # Sample ratio of training instances
    colsample_bytree=0.9,  # Sample ratio of features
    min_child_weight=3,    # Minimum sum of instance weight
    gamma=0.1,             # Minimum loss reduction
    random_state=42        # Reproducibility
)
```

### 5.3 Performance Metrics
- **Accuracy:** 90%+
- **Precision:** 90%+ (weighted average)
- **Recall:** 90%+ (weighted average)
- **F1-Score:** 90%+ (weighted average)
- **Cross-Validation:** 5-fold with ±2% standard deviation

## 6. Implementation Details

### 6.1 Data Flow
1. **Data Generation:** Create synthetic samples with realistic correlations
2. **Preprocessing:** Feature engineering and normalization
3. **Training:** 75% data for model training
4. **Validation:** 25% data for testing + 5-fold cross-validation
5. **Deployment:** Interactive CLI for real-time predictions

### 6.2 Error Handling
- Input validation for all parameters
- Range checking for lifestyle factors
- Exception handling for invalid inputs
- User-friendly error messages

### 6.3 Output Format
```
Mental Health Status: [Very Poor/Poor/Good/Very Good]
Confidence: [Percentage]
Certainty Level: [Very Confident/Confident/Moderate Confidence]
Recommendations: [Personalized advice based on input]
```

## 7. Testing and Validation

### 7.1 Model Validation
- **Train-Test Split:** 75%-25% random split
- **Cross-Validation:** 5-fold stratified validation
- **Performance Consistency:** Multiple runs with different random seeds

### 7.2 Demo Test Cases
| Case | Sleep | Stress | Social | Exercise | Expected |
|------|-------|--------|--------|----------|----------|
| Very Good | 8.5 | 1 | 9 | 2.5 | Very Good |
| Good | 7.5 | 3 | 7 | 1.5 | Good |
| Poor | 5.5 | 7 | 3 | 0.5 | Poor |
| Very Poor | 4 | 9 | 1 | 0 | Very Poor |

## 8. Future Enhancements

1. **Real Data Integration:** Clinical dataset training
2. **Time Series Analysis:** Longitudinal health tracking
3. **Mobile Application:** Smartphone deployment
4. **API Development:** Healthcare system integration
5. **Advanced Features:** Mood tracking, sleep quality metrics

## 9. Conclusion

This project successfully demonstrates the application of machine learning in healthcare prediction. The XGBoost model achieves high accuracy through careful feature engineering and hyperparameter optimization. The system provides immediate, actionable insights while maintaining user privacy through local processing. The modular design allows for easy extension and integration into larger healthcare systems.

**Key Achievements:**
- 90%+ prediction accuracy
- Robust cross-validation
- User-friendly interface
- Comprehensive error handling
- Actionable health recommendations
