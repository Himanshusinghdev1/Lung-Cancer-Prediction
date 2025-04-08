from cnnClassifier.pipeline.prediction import PredictionPipeline

# Initialize the prediction pipeline with the image file
file_path = "/Users/himanshu/Lung-Cancer-Prediction/lungaca1.jpeg"
classifier = PredictionPipeline(filename=file_path)

# Make prediction
try:
    result = classifier.predict()
    print(f"🔥 Debug Result: {result}")
except Exception as e:
    print(f"Error occurred: {str(e)}")

