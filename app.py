



# # if st.button('Predict[")

from flask import Flask,render_template,request,jsonify
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model 
import pickle
import re

# # import numpy as np

app = Flask(__name__)



# # model = load_model("model.pkl") 
model = pickle.load(open('model.pkl','rb'))

# loading
# # # with open('tokenizer.pickle', 'rb') as handle:
tokenizer = pickle.load(open('tokenizer.pickle','rb'))

label2int = {'0': 0, '1': 1}
int2label = {0: '0', 1: '1'}
SEQUENCE_LENGTH = 327

def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))
    sentence = re.sub(r'\s+', ' ', str(sentence))

    return sentence

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', str(text))

def get_predictions(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    # return int2label[np.argmax(prediction)]



@app.route("/home")
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/shop")
def shop():
    return render_template("shop.html")


@app.route("/sproduct", methods = ['GET','POST'])
def sproduct():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       reviewText = request.form.get("review")
       # getting input with name = lname in HTML form 
    #    print(get_predictions(reviewText))
       return render_template('predict.html')
    return render_template("sproduct.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

    


if __name__ == "__main__":
    app.run(debug=True)




