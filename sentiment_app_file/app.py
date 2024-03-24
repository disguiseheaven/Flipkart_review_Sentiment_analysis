from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return(render_template('home.html'))

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    text = request.form.get('review')
    model = joblib.load('sentiment_app_file\model\logistic_regression.pkl')
    if text == "":
        return render_template('home.html')
    else:
        text_review = np.array([text])
        prediction = model.predict(text_review)
        if prediction == 0:
            answer = False
        else:
            answer = True
        return (render_template('result.html', answer = answer))

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0")
