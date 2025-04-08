import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        try:
            # Load the model
            model_path = os.path.join("model", "model.h5")
            model = load_model(model_path)

            # Load and preprocess the image
            img = image.load_img(self.filename, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize

            # Predict
            prediction = model.predict(img_array)
            result = np.argmax(prediction, axis=1)

            print("Raw Prediction Output:", prediction)

            # Label map - adjust to match training labels
            label_map = {
                0: "Benign",
                1: "Normal",
                2: "Malignant"
            }

            prediction_text = label_map.get(result[0], "Unknown")
            print(f"Final Prediction: {prediction_text}")
            return {"image": prediction_text}

        except Exception as e:
            print(f"Error in PredictionPipeline: {str(e)}")
            return {"error": str(e)}