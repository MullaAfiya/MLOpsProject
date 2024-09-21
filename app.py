# from flask import Flask, request, jsonify
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import joblib

# # Load the model
# model = joblib.load("model.pkl")

# # Initialize Flask app
# app = Flask(__name__)

# # Route to make predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     rooms = np.array(data['Number_of_Rooms']).reshape(-1, 1)
    
#     prediction = model.predict(rooms)
#     output = prediction.tolist()
#     return jsonify({'Predicted Prices': output})

# # Health check route
# @app.route('/')
# def index():
#     return "ML Model API is running"

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('house_price_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    num_rooms = data['num_rooms']
    predicted_price = model.predict([[num_rooms]])[0]
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
    
    ##########
    ##To run the project 
    ## python app.py
    ## curl -X POST -H "Content-Type: application/json" -d '{ "num_rooms": 10 }' http://localhost:5000/predict
###### {"predicted_price":550.0}