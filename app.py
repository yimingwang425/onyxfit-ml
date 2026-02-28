from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import tensorflow as tf
import requests
import json

keras = tf.keras

app = Flask(__name__)
CORS(app)

model_path = os.path.dirname(os.path.abspath(__file__))

# Load the model
workout_model = keras.models.load_model(os.path.join(model_path, 'workout_model.keras'))
meal_model = keras.models.load_model(os.path.join(model_path, 'meal_model.keras'))

scaler_X = joblib.load(os.path.join(model_path, 'scaler_X.pkl'))
scaler_y_workout = joblib.load(os.path.join(model_path, 'scaler_y_workout.pkl'))
scaler_y_meal = joblib.load(os.path.join(model_path, 'scaler_y_meal.pkl'))

le_activity = joblib.load(os.path.join(model_path, 'le_activity.pkl'))
le_goal = joblib.load(os.path.join(model_path, 'le_goal.pkl'))
le_diet = joblib.load(os.path.join(model_path, 'le_diet.pkl'))
le_metabolic = joblib.load(os.path.join(model_path, 'le_metabolic.pkl'))

print("Models loaded successfully!")

GROQ_API_KEY = 'GROQ_API_KEY'
GROQ_MODEL = 'llama-3.1-8b-instant'

# 0=Sunday, 1=Monday ... 6=Saturday
WORKOUT_SCHEDULES = {
    'PPL': ['Rest', 'Push (Chest, Shoulders, Triceps)', 'Pull (Back, Biceps)',
            'Legs (Quads, Hamstrings, Glutes)', 'Push (Chest, Shoulders, Triceps)',
            'Pull (Back, Biceps)', 'Rest'],
    'UPPER_LOWER': ['Rest', 'Upper Body', 'Lower Body', 'Upper Body',
                    'Lower Body', 'Rest', 'Rest'],
    'FBW': ['Rest', 'Full Body', 'Rest', 'Full Body', 'Rest', 'Full Body', 'Rest']
}

DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def generate_weekly_meal_plan(calories, protein, carbs, fat, diet_pref, workout_type):

    print(f"\nCalling Llama 3.1 via Groq API for WEEKLY meal plan")
    print(f"   Target: {calories} kcal, {protein}g protein, {diet_pref} diet, {workout_type} split")

    schedule = WORKOUT_SCHEDULES.get(workout_type, WORKOUT_SCHEDULES['FBW'])
    rest_calories = int(calories * 0.85)

    day_targets = []
    for i in range(7):
        is_rest = (schedule[i] == 'Rest')
        cal = rest_calories if is_rest else calories
        prot = round(protein * 0.85, 1) if is_rest else protein
        label = "rest day (lighter meals)" if is_rest else "training day (high protein)"
        day_targets.append(f"  Day {i}: {cal} kcal, {prot}g protein — {label}")
    targets_text = '\n'.join(day_targets)

    prompt = f"""You are a professional chef and sports nutritionist. Create a 7-day meal plan.

Daily calorie and protein targets:
{targets_text}

Additional macros for training days: {carbs}g carbs, {fat}g fat
Diet preference: {diet_pref}

Return ONLY valid JSON. Keys "0" through "6" represent Sunday through Saturday.

Example of ONE day (Day 1):
{{
  "1": {{
    "breakfast": {{"name": "Greek Yogurt Parfait", "calories": 420, "macros": {{"p": 30, "c": 50, "f": 12}}, "ingredients": ["200g Greek yogurt", "50g granola", "100g mixed berries", "1 tbsp honey"], "recipe": ["Layer yogurt, granola and berries in a bowl", "Drizzle honey on top"]}},
    "lunch": {{"name": "Grilled Chicken Caesar Salad", "calories": 550, "macros": {{"p": 45, "c": 20, "f": 30}}, "ingredients": ["150g chicken breast", "100g romaine lettuce", "30g parmesan", "2 tbsp Caesar dressing"], "recipe": ["Grill chicken breast for 6 min each side", "Chop lettuce and toss with dressing", "Slice chicken and place on top", "Sprinkle parmesan"]}},
    "dinner": {{"name": "Baked Salmon with Roasted Vegetables", "calories": 620, "macros": {{"p": 45, "c": 40, "f": 25}}, "ingredients": ["180g salmon fillet", "200g sweet potato", "100g broccoli", "1 tbsp olive oil"], "recipe": ["Preheat oven to 200C", "Season salmon and bake 15 min", "Cube sweet potato and roast 25 min", "Steam broccoli 4 min"]}},
    "snack": {{"name": "Peanut Butter Banana Smoothie", "calories": 300, "macros": {{"p": 20, "c": 35, "f": 10}}, "ingredients": ["1 banana", "2 tbsp peanut butter", "200ml milk", "1 scoop protein powder"], "recipe": ["Blend all ingredients until smooth"]}}
  }}
}}

Now generate ALL 7 days ("0" through "6") following that exact format.

STRICT RULES:
1. Every "name" must be a real dish name (e.g. "Teriyaki Chicken Bowl", "Mushroom Risotto"). NEVER use generic names like "Day 1 Breakfast" or "Push Lunch".
2. All 7 days must have completely different meals — 28 unique dishes total.
3. "ingredients" and "recipe" must be arrays of plain strings.
4. Each day's 4 meals must add up to approximately that day's calorie target.
5. Rest day meals should be lighter with more vegetables; training day meals higher in protein and carbs."""

    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': GROQ_MODEL,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a professional chef and sports nutritionist. Return only valid JSON. Every meal name must be a real dish name like "Chicken Stir-Fry" or "Avocado Toast" — never generic labels. Use plain strings for ingredients and recipe arrays.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.7,
                'max_tokens': 4096,
                'response_format': {'type': 'json_object'}
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']

            clean_content = content.strip()
            if clean_content.startswith('```'):
                lines = clean_content.split('\n')
                clean_content = '\n'.join(lines[1:-1])

            weekly_plan = json.loads(clean_content)

            days_found = [k for k in weekly_plan.keys() if k in ['0','1','2','3','4','5','6']]
            print(f"Weekly meal plan generated via Groq ({len(days_found)} days):")
            for d in sorted(days_found):
                day_meals = weekly_plan[d]
                if day_meals and isinstance(day_meals, dict):
                    bname = day_meals.get('breakfast', {}).get('name', '?')
                    print(f"   Day {d} ({DAY_NAMES[int(d)]}): {bname} ...")

            return weekly_plan
        else:
            print(f"Groq API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("timed out (30s)")
        return None
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Groq API.")
        return None
    except json.JSONDecodeError as e:
        print(f"error: {e}")
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Unexpected error: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    print("Received prediction request")

    data = request.get_json()

    age = data['age']
    height = data['heightCm']
    weight = data['weightKg']
    activity = data['activityLevel']
    goal = data['goal']
    diet = data['dietPref']
    metabolic = data['metabolicProfile']

    print(f"User: {age}y, {height}cm, {weight}kg, {activity}, {goal}, {diet}")

    activity_code = le_activity.transform([activity])[0]
    goal_code = le_goal.transform([goal])[0]
    diet_code = le_diet.transform([diet])[0]
    metabolic_code = le_metabolic.transform([metabolic])[0]

    user_input = np.array([[age, height, weight, activity_code, goal_code, diet_code, metabolic_code]])
    user_input_scaled = scaler_X.transform(user_input)

    print("Running Keras predictions")
    workout_pred = workout_model.predict(user_input_scaled, verbose=0)
    meal_pred = meal_model.predict([user_input_scaled, workout_pred], verbose=0)

    workout_result = scaler_y_workout.inverse_transform(workout_pred)
    meal_result = scaler_y_meal.inverse_transform(meal_pred)

    workout_type_num = int(round(workout_result[0][1]))
    workout_type_num = max(0, min(2, workout_type_num))
    workout_type = {0: 'FBW', 1: 'UPPER_LOWER', 2: 'PPL'}[workout_type_num]

    calories = int(meal_result[0][0])
    protein = round(float(meal_result[0][1]), 1)
    carbs = round(float(meal_result[0][2]), 1)
    fat = round(float(meal_result[0][3]), 1)

    print(f"Keras done: {calories} kcal, {protein}g protein, {workout_type}")

    weekly_meal_plan = generate_weekly_meal_plan(
        calories, protein, carbs, fat, diet, workout_type
    )

    response = {
        'caloriesKcal': calories,
        'proteinG': protein,
        'carbsG': carbs,
        'fatG': fat,
        'workoutIntensity': round(float(workout_result[0][0]), 2),
        'workoutType': workout_type
    }

    if weekly_meal_plan:
        response['weeklyMealPlan'] = weekly_meal_plan
        print("Response includes weeklyMealPlan (7 days)")
    else:
        print("Groq failed")

    return jsonify(response), 200


@app.route('/api/insight', methods=['POST'])
def generate_insight():
    data = request.get_json() or {}

    prompt = f"""You are a concise fitness coach. Based on this user's data, give ONE short personalized insight (1-2 sentences max).

User context:
- Today's workout: {data.get('workout', 'Unknown')}
- Daily calorie target: {data.get('calories', 'Unknown')} kcal
- Workout program: {data.get('workoutType', 'Unknown')}
- Current mood: {data.get('mood', 'Not logged')}
- Water intake: {data.get('water', 0)} cups today
- Fitness goal: {data.get('goal', 'Unknown')}

Rules:
- Be specific and actionable, not generic
- Reference their actual data (mood, workout type, water, etc.)
- Keep it under 30 words
- Do NOT use quotes or markdown
- Just return the plain text insight, nothing else"""

    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': GROQ_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.8,
                'max_tokens': 100
            },
            timeout=10
        )

        if resp.status_code == 200:
            result = resp.json()
            insight = result['choices'][0]['message']['content'].strip()
            return jsonify({'insight': insight}), 200
        else:
            return jsonify({'insight': 'Stay consistent with your plan today. Every session counts.'}), 200

    except Exception as e:
        print(f"Insight generation error: {e}")
        return jsonify({'insight': 'Stay consistent with your plan today. Every session counts.'}), 200


if __name__ == '__main__':
    print("\nFlask ML Service Starting")
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
