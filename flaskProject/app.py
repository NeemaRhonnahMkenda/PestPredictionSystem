

from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

df = pd.read_csv('/Users/neemamkenda/PycharmProjects/flaskProject/datasets/MOCK_WEATHER_DATA .csv')

# Extract features and target variable
X = df.drop(['log_id', 'date', 'pest_infestation'], axis=1)
y = df['pest_infestation']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)

# Define categorical features
categorical_features = ['crop_type']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42)),
])

# Model Training
pipeline.fit(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        crop_type = request.form['crop_type']
        temperature = float(request.form['temperature'])
        date = pd.to_datetime(request.form['date'])
        rainfall = float(request.form['rainfall'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        soil_moisture = float(request.form['soil_moisture'])

        # Create a DataFrame from user input
        user_input = {
            'date': [date],
            'temperature': [temperature],
            'humidity': [humidity],
            'rainfall': [rainfall],
            'wind_speed': [wind_speed],
            'crop_type': [crop_type],
            'soil_moisture': [soil_moisture],
        }

        user_input_df = pd.DataFrame(user_input)

        # Convert 'date' to datetime
        user_input_df['date'] = pd.to_datetime(user_input_df['date'])

        # Ensure the 'crop_type' column is present in the user input DataFrame
        user_input_df['crop_type'] = user_input_df['crop_type'].astype('category')

        # Feature Extraction for User Input
        user_input_features = pipeline.named_steps['preprocessor'].transform(user_input_df)

        # Prediction for User Input
        user_probabilities = pipeline.named_steps['classifier'].predict_proba(user_input_features)

        # Assuming the second column of user_probabilities corresponds to the positive class (pest infestation)
        pest_likelihood = user_probabilities[0][1] * 100

        # Set the threshold for classification (e.g., 50%)
        threshold = 50

        # Check the likelihood and display HIGH or LOW
        if pest_likelihood > threshold:
            pest_infestation = "HIGH"
            # Get the entered crop type
            entered_crop_type = user_input_df['crop_type'].iloc[0]

            # Check if the entered crop type is in the pest_info dictionary
            pest_info = {
                'tomatoes': {
                    'pests': '\n 1. Aphids,\n 2. Whiteflies,\n 3. Hornworms \n',
                    'mitigation': {
                        'A: early_stage': 'Use insecticidal soap, plant companion crops like marigolds \n',
                        'B: mid_stage': 'Regularly inspect plants, use neem oil, introduce beneficial insects \n',
                        'C: late_stage': 'Harvest ripe tomatoes promptly, remove plant debris, practice crop rotation \n'
                    }
                },
                'maize': {
                    'pests': '\n 1. Armyworms, \n 2. Borers, \n 3. Cutworms \n \n',
                    'mitigation': {
                        'A: early_stage': 'Practice clean cultivation, use biological control methods \n',
                        'B: mid_stage': 'Implement crop rotation, use pheromone traps \n',
                        'C: late_stage': 'Harvest maize promptly, destroy crop residues, use resistant varieties \n'
                    }
                },
                'potatoes': {
                    'pests': '\n 1. Colorado Potato Beetles,\n 2. Aphids,\n 3. Flea Beetles \n',
                    'mitigation': {
                        'A: early_stage': 'Remove and destroy infested leaves, use insecticidal soap \n',
                        'B: mid_stage': 'Rotate crops, plant potatoes away from tomatoes, peppers \n',
                        'C: late_stage': 'Harvest potatoes when mature, remove plant debris, practice crop rotation \n'
                    }
                },
                'beans': {
                    'pests': '\n 1. Aphids, \n 2. Mexican Bean Beetles, \n 3. Thrips \n',
                    'mitigation': {
                        'A: early_stage': 'Use insecticidal soap, encourage natural predators \n',
                        'B: mid_stage': 'Introduce ladybugs, handpick beetles, rotate crops \n',
                        'C: late_stage': 'Harvest beans regularly, remove debris, practice crop rotation \n'
                    }
                },
                'wheat': {
                    'pests': '\n 1. Aphids, \n 2. Hessian Fly, \n 3. Armyworms \n',
                    'mitigation': {
                        'A: early_stage': 'Select resistant wheat varieties, monitor for aphids \n',
                        'B: mid_stage': 'Practice crop rotation, use insecticidal soap sparingly \n',
                        'C: late_stage': 'Harvest wheat promptly, destroy crop residues, practice clean cultivation \n'
                    }
                }
            }

            pest_info = pest_info.get(entered_crop_type, {})
        else:
            pest_infestation = "LOW"
            pest_info = {}

        return render_template('result.html', pest_likelihood=pest_likelihood, pest_infestation=pest_infestation, pest_info=pest_info, user_input_df=user_input_df)


if __name__ == '__main__':
    app.run(debug=True)
