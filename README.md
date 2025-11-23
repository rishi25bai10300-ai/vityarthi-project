# Mental Health Predictor

A machine learning tool that predicts mental health status based on lifestyle factors using XGBoost classification.

## Features

- Predicts mental health status (Very Poor, Poor, Good, Very Good)
- Uses 11 lifestyle factors including sleep, stress, exercise, and social interaction
- Provides confidence scores and personalized recommendations
- Interactive command-line interface
- Cross-validated model with 90%+ accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the predictor:

```bash
python mental_health.py
```

The program will:
1. Train the model on generated data
2. Show performance metrics
3. Display demo predictions
4. Prompt for your personal data

### Quick Prediction Function

```python
from mental_health import quick_predict

# Basic prediction with 4 core factors
result = quick_predict(sleep_hours=8, stress_level=3, social_interaction=7, exercise_hours=1.5)
print(result)
```

## Input Parameters

- **Sleep hours** (3-12): Hours of sleep per night
- **Stress level** (0-10): Self-reported stress (0=none, 10=extreme)
- **Social interaction** (0-10): Quality of social connections
- **Exercise hours** (0-4): Daily physical activity
- **Diet quality** (0-10): Nutrition quality [optional]
- **Work hours** (4-16): Daily work hours [optional]
- **Screen time** (1-24): Daily screen exposure [optional]
- **Meditation hours** (0-3): Daily mindfulness practice [optional]
- **Age** (18-65): Your age [optional]

## Model Performance

- **Accuracy**: ~90%
- **Cross-validation**: 5-fold validation
- **Algorithm**: XGBoost with optimized hyperparameters
- **Training data**: 100,000 synthetic samples with realistic correlations

## Key Factors

The model identifies these as most important for mental health:
1. Sleep quality and duration
2. Stress management
3. Social connections
4. Physical exercise
5. Work-life balance

## Requirements

- Python 3.7+
- scikit-learn >= 1.3.0
- xgboost >= 1.7.0
- numpy >= 1.21.0
