import os
import json
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get('MODEL_PATH', 'trained_model.pkl')
PIPELINE_PATH = os.environ.get('PIPELINE_PATH', 'csv_preprocessor.pkl')
DATA_PATH = os.environ.get('DATA_PATH', 'prediction_data.csv')

def load_model_and_data():
    """
    Load the model, pipeline, and dataset from the specified paths.

    Returns:
        tuple: A tuple containing the loaded model, pipeline, and dataframe.
    """
    try:
        model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        df = pd.read_csv(DATA_PATH)
        return model, pipeline, df
    except Exception as e:
        logging.error(f"Error loading model, pipeline, or data: {e}", exc_info=True)
        raise

try:
    model, pipeline, df = load_model_and_data()
except Exception as e:
    logging.critical(f"Failed to load model, pipeline, or data: {e}", exc_info=True)
    raise

def create_app():
    """
    Create and configure the Flask application.

    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Predict the probability of default for a given SK_ID_CURR.

        Returns:
            Response: A JSON response containing the prediction and additional features.
        """
        try:
            data = request.get_json()
            sk_id_curr = data.get('SK_ID_CURR')

            if sk_id_curr is None:
                return jsonify({'error': 'Missing SK_ID_CURR in request; number should be from 1 to 30752'}), 400

            sk_id_curr = int(sk_id_curr)
            user_data = df[df['SK_ID_CURR'] == sk_id_curr]

            if user_data.empty:
                return jsonify({'error': 'SK_ID_CURR not found'}), 404

            user_data_transformed = pipeline.transform(user_data)
            probability = model.predict_proba(user_data_transformed)[:, 1][0]
            probability_percentage = f"{probability * 100:.2f}%"

            additional_features = user_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_MEAN',
                                             'AMT_CREDIT', 'CREDIT_ANNUITY_RATIO', 'AGE', 'AMT_GOODS_PRICE',
                                             'DAYS_EMPLOYED']].round(3).to_dict(orient='records')[0]

            response = {
                'SK_ID_CURR': sk_id_curr,
                'default_probability': probability_percentage,
                **additional_features
            }

            return jsonify(response)

        except Exception as e:
            logging.error(f"Error in /predict endpoint: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8080)