import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelRF1.pkl', 'rb'))
#model1 = pickle.load(open('model1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/lap')
def lap():
    return render_template('Lap.html')

@app.route('/predictmob', methods=['POST'])
def predictmob():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    crop = ('rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
            'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee')
    out=crop[output]

    return render_template('index.html', prediction_text='Sugesstion to  grow {} crop'.format(out))


@app.route('/predictlap', methods=['POST'])
def predictlap():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('lap.html', prediction_text='Laptop Price should be Rs {}'.format(output))


"""
@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    """


if __name__ == "__main__":
    app.run()

