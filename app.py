from flask import Flask, render_template, request
from cnnClassifier.pipeline.prediction import PredictionPipeline
import os

#app name is app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        image_file = request.files["imagefile"]
        if image_file:
            # Save image
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)

            # Predict
            pipeline = PredictionPipeline(image_path)
            result = pipeline.predict()

            if "error" in result:
                prediction = f"Error: {result['error']}"
            else:
                prediction = result["image"]

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    
    app.run(debug=True)