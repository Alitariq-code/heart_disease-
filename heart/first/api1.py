from keras.models import load_model
from joblib import load
import numpy as np
from scipy.stats import mode
import os
knn_model = load(r'C:\Users\miana\Desktop\django_heart_1\heart_project\heart\first\knn_model.joblib')
base_path = 'C:/Users/miana/Desktop/django_heart_1/heart_project/heart/first/'
rf_model_path = os.path.join(base_path, 'rf_model.joblib')

rf_model_path = os.path.join(base_path, 'rf_model.joblib')
dtc_model_path = os.path.join(base_path, 'dtc_model.joblib')
rf_model = load(rf_model_path)
dtc_model = load(dtc_model_path)

# Load Keras models from .h5 files
model_cnn_path = os.path.join(base_path, 'model_cnn1.h5')
model_lstm_path = os.path.join(base_path, 'model_cnn2.h5')
model_gru_path = os.path.join(base_path, 'model_cnn3.h5')
model_cnn = load_model(model_cnn_path)
model_lstm = load_model(model_lstm_path)
model_gru = load_model(model_gru_path)
scaler_path = os.path.join(base_path, 'scaler.joblib')
scaler = load(scaler_path) 


def load_models_and_vote(input_features):
    # Load sklearn models
    

    
    # Prepare input
    input_scaled = scaler.transform([input_features])

    # Predict with sklearn models
    knn_predictions = knn_model.predict(input_scaled)
    rf_predictions = rf_model.predict(input_scaled)
    dtc_predictions = dtc_model.predict(input_scaled)

    # Predict with Keras models and convert probabilities to binary predictions
    threshold = 0.5
    cnn_predictions = (model_cnn.predict(input_scaled)[:, 0] < threshold).astype(int)
    lstm_predictions = (model_lstm.predict(np.expand_dims(input_scaled, axis=-1))[:, 0] < threshold).astype(int)
    gru_predictions = (model_gru.predict(np.expand_dims(input_scaled, axis=-1))[:, 0] < threshold).astype(int)

     # Combine all predictions
    combined_predictions, _ = mode([knn_predictions, rf_predictions, dtc_predictions, cnn_predictions, lstm_predictions, gru_predictions])

    return combined_predictions
    print(combined_predictions)
# Example usage
input_features = [53, 1, 1, 206, 92, 215]  # Example input
prediction = load_models_and_vote(input_features)
print(prediction)