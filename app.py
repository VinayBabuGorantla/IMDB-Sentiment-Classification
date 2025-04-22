import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = only errors

from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "GET":
        return render_template("home.html")
    else:
        review = request.form.get("review_text")
        pipeline = PredictPipeline()
        sentiment, score = pipeline.predict(review)
        return render_template("home.html", review=review, sentiment=sentiment, score=round(score, 4))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
