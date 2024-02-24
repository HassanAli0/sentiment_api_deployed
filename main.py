from flask import Flask, render_template, request, jsonify
import joblib
import os

model_file_path = 'sentimentModel2.pkl'
model, cv = joblib.load(model_file_path)

from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
stemmer = PorterStemmer()

from nltk.corpus import stopwords

# Download the stopwords corpus if you haven't already done so
nltk.download('stopwords')

english_stopwords = stopwords.words('english')

def clean_text(eng_text):
    eng_text = eng_text.lower()
    eng_text = nltk.word_tokenize(eng_text)
    eng_text = [t for t in eng_text if len(t) > 1]
    eng_text = [stemmer.stem(word) for word in eng_text if word not in english_stopwords]
    eng_text = ' '.join(eng_text)
    return eng_text



app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/comment', methods=['POST', 'GET'])
def comment():
    data = request.json
    text = data['text']
    text=clean_text(text)
    X_sentence = cv.transform([text])
    sentiment = ''
    if(model.predict(X_sentence)==0):
        sentiment = "Negative Sentence"
    elif(model.predict(X_sentence)==4):
        sentiment = "Positive Sentence"

    return jsonify({"sentiment": sentiment})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
