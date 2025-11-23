import random
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


def generate_enhanced_data(n_samples=100000):
    data = []
    labels = []

    for _ in range(n_samples):
        age = random.randint(18, 65)
        age_factor = 1 - (abs(age - 35) / 50)

        base_health = random.uniform(0, 1) * age_factor

        if base_health > 0.7:
            sleep = np.random.normal(8, 0.5)
            stress = random.randint(0, 4)
            social = random.randint(6, 10)
            exercise = np.random.gamma(2, 0.5)
            diet_quality = random.randint(7, 10)
            work_hours = np.random.normal(7.5, 1)
            screen_time = np.random.normal(4, 1)
            meditation = np.random.exponential(0.8)
        elif base_health < 0.3:
            sleep = np.random.normal(5, 0.8)
            stress = random.randint(6, 10)
            social = random.randint(0, 4)
            exercise = np.random.exponential(0.2)
            diet_quality = random.randint(0, 4)
            work_hours = np.random.normal(11, 1.5)
            screen_time = np.random.normal(9, 1.5)
            meditation = np.random.exponential(0.1)
        else:
            sleep = np.random.normal(6.5, 1)
            stress = random.randint(3, 7)
            social = random.randint(3, 7)
            exercise = np.random.gamma(1.5, 0.4)
            diet_quality = random.randint(4, 7)
            work_hours = np.random.normal(8.5, 1.2)
            screen_time = np.random.normal(6.5, 1.5)
            meditation = np.random.exponential(0.4)

        sleep = np.clip(sleep, 3, 12)
        exercise = np.clip(exercise, 0, 4)
        work_hours = np.clip(work_hours, 4, 16)
        screen_time = np.clip(screen_time, 1, 24)
        meditation = np.clip(meditation, 0, 3)

        sleep_stress_ratio = sleep / (stress + 1)
        work_life_balance = (exercise + social/2) - (work_hours / 3)
        age_normalized = (age - 18) / 47

        score = (
            (sleep/12 * 0.25) +
            ((10-stress)/10 * 0.25) +
            (social/10 * 0.2) +
            (exercise/4 * 0.15) +
            (diet_quality/10 * 0.1) +
            (meditation/3 * 0.05)
        )

        if score > 0.75:
            label = 3
        elif score > 0.6:
            label = 2
        elif score > 0.4:
            label = 1
        else:
            label = 0

        features = [sleep, stress, social, exercise, diet_quality,
                    work_hours, screen_time, meditation, age_normalized,
                    sleep_stress_ratio, work_life_balance]

        data.append(features)
        labels.append(label)

    return np.array(data), np.array(labels)


print("Generating high-quality dataset...")
X, y = generate_enhanced_data(100000)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)


model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("\n MODEL PERFORMANCE:")
print(f"Accuracy: {accuracy:.1%}")
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
print(f"F1-Score: {f1:.1%}")
print(f"Cross-validation: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

feature_names = ['Sleep', 'Stress', 'Social', 'Exercise', 'Diet',
                 'Work Hours', 'Screen Time', 'Meditation', 'Age',
                 'Sleep/Stress Ratio', 'Work-Life Balance']
importance = model.feature_importances_
sorted_features = sorted(zip(feature_names, importance), key=lambda x: x[1],
                         reverse=True)

print("\n TOP FACTORS AFFECTING MENTAL HEALTH:")
for name, imp in sorted_features[:5]:
    print(f"• {name}: {imp:.3f}")


def validate_input(value, min_val, max_val, name):
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return value


def predict_mental_health_enhanced(sleep_hours, stress_level,
                                   social_interaction, exercise_hours,
                                   diet_quality, work_hours,
                                   screen_time, meditation_hours, age):

    validate_input(sleep_hours, 3, 12, "Sleep hours")
    validate_input(stress_level, 0, 10, "Stress level")
    validate_input(social_interaction, 0, 10, "Social interaction")
    validate_input(exercise_hours, 0, 4, "Exercise hours")
    validate_input(diet_quality, 0, 10, "Diet quality")
    validate_input(work_hours, 4, 16, "Work hours")
    validate_input(screen_time, 1, 24, "Screen time")
    validate_input(meditation_hours, 0, 3, "Meditation hours")
    validate_input(age, 18, 65, "Age")

    sleep_stress_ratio = sleep_hours / (stress_level + 1)
    work_life_balance = ((exercise_hours + social_interaction) -
                         (work_hours / 2))
    age_normalized = (age - 18) / 47

    features = np.array([[sleep_hours, stress_level, social_interaction,
                          exercise_hours, diet_quality, work_hours,
                          screen_time, meditation_hours, age_normalized,
                          sleep_stress_ratio, work_life_balance]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    status_map = {0: "Very Poor", 1: "Poor", 2: "Good", 3: "Very Good"}
    status = status_map[prediction]
    confidence = max(probability) * 100

    if confidence > 90:
        certainty = "Very Confident"
    elif confidence > 75:
        certainty = "Confident"
    else:
        certainty = "Moderate Confidence"

    return f"Mental Health Status: {status} ({certainty}: {confidence:.1f}%)"


def quick_predict(sleep_hours, stress_level, social_interaction,
                  exercise_hours):
    return predict_mental_health_enhanced(
        sleep_hours, stress_level, social_interaction, exercise_hours,
        diet_quality=7,
        work_hours=8,
        screen_time=6,
        meditation_hours=0.2,
        age=30
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("ENHANCED MENTAL HEALTH PREDICTOR")
    print("="*50)

    print("\n DEMO PREDICTIONS:")
    print("Very good case:", quick_predict(8.5, 1, 9, 2.5))
    print("Good case:", quick_predict(7.5, 3, 7, 1.5))
    print("Poor case:", quick_predict(5.5, 7, 3, 0.5))
    print("Very poor case:", quick_predict(4, 9, 1, 0))

    print("\n" + "="*50)
    print(" ENTER YOUR DATA:")

    try:
        sleep = float(input("Sleep hours (3-12): "))
        stress = int(input("Stress level (0-10): "))
        social = int(input("Social interaction (0-10): "))
        exercise = float(input("Exercise hours (0-4): "))

        print("\n OPTIONAL (press Enter for defaults):")
        diet = input("Diet quality (0-10) [default: 7]: ")
        work = input("Work hours (4-16) [default: 8]: ")
        screen = input("Screen time (1-24) [default: 6]: ")
        meditation = input("Meditation hours (0-3) [default: 0.2]: ")
        age = input("Age (18-65) [default: 30]: ")

        diet = float(diet) if diet else 7
        work = float(work) if work else 8
        screen = float(screen) if screen else 6
        meditation = float(meditation) if meditation else 0.2
        age = int(age) if age else 30

        result = predict_mental_health_enhanced(sleep, stress, social,
                                                exercise, diet, work, screen,
                                                meditation, age)

        print(f"\n RESULT: {result}")

        print("\n RECOMMENDATIONS:")
        if sleep < 7:
            print("• Try to get 7-9 hours of sleep")
        if stress > 6:
            print("• Consider stress management techniques")
        if social < 5:
            print("• Increase social interactions")
        if exercise < 1:
            print("• Add 30+ minutes of daily exercise")

    except ValueError as e:
        print(f" Error: {e}")
    except KeyboardInterrupt:
        print("\n Goodbye!")
