from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

# Initialize ClientApp class
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        file_path = "inputImage.jpg"
        file.save(file_path)

        # Ensure the file is saved correctly
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not saved correctly.")

        # Update classifier filename
        clApp.classifier.filename = file_path

        print(f"🔥 File saved at: {file_path}")

        # Get prediction result
        result = clApp.classifier.predict()

        if not result or 'error' in result[0]:
            raise Exception(result[0]['error'])

        prediction_text = result[0]["image"]
        print(f"✅ Prediction Result: {prediction_text}")

        # Map predictions to detailed messages
        class_labels = {
            "Benign": "The tumor is benign (non-cancerous). However, medical consultation is recommended.",
            "Malignant": "The tumor is malignant (cancerous). Immediate medical attention is advised.",
            "No Tumor": "No tumor detected. However, regular medical check-ups are encouraged."
        }

        # Get response message
        response_message = class_labels.get(prediction_text, "Unknown classification result.")

        # Prepare final JSON response
        response_data = {
            "prediction": prediction_text,
            "message": response_message
        }
        
        print(f"🔍 Full Response: {response_data}")  # Debugging log
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
