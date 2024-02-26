# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along. A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data.

## Neural Network Model

![image](https://github.com/Madhav005/basic-nn-model/assets/110885274/836c168f-0904-4246-b326-9b3e6a4c79c0)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Madhavan M
### Register Number: 212222040089
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('sampledl').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})
dataset1.head()
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([Dense(units=3,input_shape=[1]), Dense(units=3), Dense(units=1)])
ai_brain.compile(optimizer="rmsprop", loss="mae")
ai_brain.fit(X_train1, y_train, epochs=2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```

## Dataset Information

![Screenshot 2024-02-26 231340](https://github.com/Madhav005/basic-nn-model/assets/110885274/7ec3125d-d475-4ed5-a3e0-5e66f24f1380)

## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-02-26 231611](https://github.com/Madhav005/basic-nn-model/assets/110885274/adfea0f3-11bb-4650-a25f-e7d7c13fa012)

### Test Data Root Mean Squared Error

![image](https://github.com/Madhav005/basic-nn-model/assets/110885274/f3408d4b-0f67-4b3d-a119-84bc6bf24189)

### New Sample Data Prediction

 ![image](https://github.com/Madhav005/basic-nn-model/assets/110885274/cf5e6ace-57b2-42df-84ab-420bd6dd64a1)

## RESULT

Thus, a neural network regression model for the dataset is created and successfully executed.
