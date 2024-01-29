# Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.models import model_from_json
import seaborn as sns 
import matplotlib.pyplot as plt 
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('Admission_Predict_Ver1.1.csv')

df.drop(columns=['Serial No.'],inplace=True)


X = df.drop('Chance of Admit ',axis=1)
y = df['Chance of Admit ']

# Train Test Split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scale data 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Load the model architecture from JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the trained weights into the model
loaded_model.load_weights("model_weights.h5")




# Get user input from the console
gre_score = float(input("Enter GRE score: "))
toefl_score = float(input("Enter TOEFL score: "))
university_rating = float(input("Enter university rating: "))
sop = float(input("Enter SOP score: "))
lor = float(input("Enter LOR score: "))
cgpa = float(input("Enter CGPA: "))
research = float(input("Enter research experience (0 or 1): "))

# Make a prediction using the loaded model
input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
scaled_input_data = sc.transform(input_data)  # Scale the input data
prediction = loaded_model.predict(scaled_input_data)

# Print the predicted chance of admit
print(f"Predicted Chance of Admit: {prediction[0][0]}")


print(prediction)
