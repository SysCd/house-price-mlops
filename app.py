from flask import Flask, request, jsonify
import pandas as pd
import mlflow.sklearn
import logging

app = Flask(__name__)
logging.basicConfig(filename='model.log', level=logging.INFO)
model = mlflow.sklearn.load_model("models:/house-price-model/1")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        X = pd.DataFrame([data], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
        prediction = model.predict(X)[0]
        logging.info(f"Prediction: {prediction} for input: {data}")
        return jsonify({'prediction': prediction})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
    
    