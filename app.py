from flask import Flask, render_template, request
import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('Admission_Predict_Ver1.1.csv')

df.drop(columns=['Serial No.'],inplace=True)


X = df.drop('Chance of Admit ',axis=1)
y = df['Chance of Admit ']

# Train Test Split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


app = Flask(__name__)

# Load the model architecture from JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load the trained weights into the model
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.h5")

# Create a MinMaxScaler for scaling input data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gre_score = float(request.form['gre_score'])
        toefl_score = float(request.form['toefl_score'])
        university_rating = float(request.form['university_rating'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])
        research = float(request.form['research'])

        input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
        scaled_input_data = scaler.transform(input_data)  # Scale the input data
        prediction = loaded_model.predict(scaled_input_data)[0][0]

        print(prediction)
        # Render the template with the prediction
        return render_template('index.html', prediction =f'{prediction:.2f}')  # Format the prediction to two decimal places

if __name__ == '__main__':
    app.run(debug=True)
