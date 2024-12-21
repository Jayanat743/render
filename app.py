from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and features
with open('C:\Users\jayant\Downloads\Project\models\model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the model and feature names
rf_model = model_data['model']
feature_names = model_data['feature_names']
categorical_options = model_data.get('categorical_options', {})  # Optional dropdown values for categorical columns

@app.route('/')
def home():
    # Render the HTML page for user input
    return render_template('index.html', 
                           feature_names=feature_names, 
                           categorical_options=categorical_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Prepare input features
        input_features = []
        for feature in feature_names:
            if feature in categorical_options:  # Handle categorical features
                input_features.append(int(request.form[feature]))
            else:  # Handle numerical features
                input_features.append(float(request.form[feature]))
        
        # Convert input to numpy array
        input_array = np.array(input_features).reshape(1, -1)

        # Predict using the loaded model
        prediction = rf_model.predict(input_array)
        result = round(prediction[0], 2)

        return render_template('index.html', 
                               feature_names=feature_names, 
                               categorical_options=categorical_options, 
                               prediction_text=f'Predicted Laptop Price: ₹{result}')
    except Exception as e:
        return render_template('index.html', 
                               feature_names=feature_names, 
                               categorical_options=categorical_options, 
                               error_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
