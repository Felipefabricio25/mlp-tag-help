import json
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from paho.mqtt import client as mqtt_client
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt



broker = 'broker.emqx.io'
port = 1883
topic = "cmd_vel_mqtt"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'
username = 'emqx'
password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, result):

    msg_count = 0
    msg = f"{result[0][0]} + {result[0][1]}"
    print(f"result - {msg} ")
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print("Send " + str(msg) + " to topic " + str(topic))
    else:
        print("Failed to send message to topic" + str(topic))
    msg_count += 1

x1 = []
x2 = []
x3 = []
y = []

stuffie = 0

with open("FinalMatter.json", "r+") as fi:
    datajson = json.load(fi)

    for things in datajson:
        stuffie += 1
        #print(things)
        if things[4] == 393:
        
            print(things)

            velues = things[3].split(" + ")

            # if float(velues[0]) + float(velues[1]) != 0:
            
            if True:
                y.append([float(velues[0]), float(velues[1])])

                x1.append([things[0], things[1], things[2], things[5], things[6], things[7]])
                x2.append([things[0], things[1], things[2]])
                x3.append([things[5], things[6], things[7]])
            else:
                pass

with open("FinalMatter2.json", "r+") as fi:
    datajson = json.load(fi)

    for things in datajson:
        stuffie += 1
        #print(things)
        if things[4] == 393:

            velues = things[3].split(" + ")

            # if float(velues[0]) + float(velues[1]) != 0:  

            print(velues)
            
            if True:
                y.append([float(velues[0]), float(velues[1])])

                x1.append([things[0], things[1], things[2], things[5], things[6], things[7]])
                x2.append([things[0], things[1], things[2]])
                x3.append([things[5], things[6], things[7]])
            else:
                pass

with open("FInalMatter3.json", "r+") as fi:
    datajson = json.load(fi)

    for things in datajson:
        stuffie += 1
        #print(things)
        if things[4] == 393:

            velues = things[3].split(" + ")

            # if float(velues[0]) + float(velues[1]) != 0:
            
            if True:
                y.append([float(velues[0]), float(velues[1])])

                x1.append([things[0], things[1], things[2], things[5], things[6], things[7]])
                x2.append([things[0], things[1], things[2]])
                x3.append([things[5], things[6], things[7]])
            else:
                pass

with open("FinalMatter4.json", "r+") as fi:
    datajson = json.load(fi)

    for things in datajson:
        stuffie += 1
        #print(things)
        if things[4] == 393:

            velues = things[3].split(" + ")

            # if float(velues[0]) + float(velues[1]) != 0:
            
            if True:
                y.append([float(velues[0]), float(velues[1])])

                x1.append([things[0], things[1], things[2], things[5], things[6], things[7]])
                x2.append([things[0], things[1], things[2]])
                x3.append([things[5], things[6], things[7]])
            else:
                pass




#print((veltime))

print("---")
      
#print(arucotime)

X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


regr = MLPRegressor(hidden_layer_sizes=(100,100), random_state=1, activation='logistic', max_iter=500).fit(X_train, y_train)


# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'solver': ['lbfgs', 'sgd', 'adam']
# }

# # Define GridSearchCV object
# grid = GridSearchCV(regr, param_grid=param_grid, cv=5, n_jobs=-1)

# # Fit the model
# grid.fit(x1, y)

# # Print best parameters and score
# print("Best parameters: ", grid.best_params_)
# print("Best score: ", grid.best_score_)


#---

X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y, test_size=0.2, random_state=1)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train2, y_train2, test_size=0.2, random_state=42)

regr2 = MLPRegressor(hidden_layer_sizes=(100,100), random_state=1, activation='logistic', max_iter=500).fit(X_train2, y_train2)

#---

X_train3, X_test3, y_train3, y_test3 = train_test_split(x3, y, test_size=0.2, random_state=1)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train3, y_train3, test_size=0.2, random_state=42)


regr3 = MLPRegressor(hidden_layer_sizes=(100,100), random_state=1, activation='logistic', max_iter=500  ).fit(X_train3, y_train3)

#---

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(3,), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))) 
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(2, activation='linear'))

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train3, y_train3, epochs=1, batch_size=32)

#---

model2 = Sequential()
model2.add(Dense(512, activation='relu', input_shape=(3,), kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))) 
model2.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dense(2, activation='linear'))

optimizer = Adam(lr=0.0001)
model2.compile(optimizer=optimizer, loss='mean_squared_error')

model2.fit(X_train2, y_train2, epochs=1, batch_size=32)

#---

model3 = Sequential()
model3.add(Dense(512, activation='relu', input_shape=(6,), kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.5))
model3.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.5))
model3.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.5))
model3.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.5))
model3.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.5))
model3.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))) 
model3.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.5))
model3.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dense(2, activation='linear'))

optimizer = Adam(lr=0.0001)
model3.compile(optimizer=optimizer, loss='mean_squared_error')

model3.fit(X_train, y_train, epochs=1, batch_size=32)

#---

print(X_test[:2])
print(regr.predict(X_test[:2]))

print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
acc = metrics.r2_score(y_test, y_pred)
print(acc)

scores = cross_val_score(regr, x1, y, cv=5, scoring='neg_mean_squared_error')

# Print the average score and its standard deviation
print("Cross-validation scores:", scores)
print("Average score:", scores.mean())
print("Standard deviation:", scores.std())

# a = accuracy_score(y_test, y_pred)

# print(a)

# print(" Acc: " + str(accuracy_score(y_test, y_pred))")

print(y_test, y_pred)

nums = list(range(0,len(y_test)))

plt.plot(nums, y_test, color="red")
plt.plot(nums, y_pred, color="yellow")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print("---")


print(X_test2[:2])
print(regr2.predict(X_test2[:2]))

print(regr2.score(X_test2, y_test2))

y_pred2 = regr2.predict(X_test2)
acc = metrics.r2_score(y_test2, y_pred2)
print(acc)

scores = cross_val_score(regr2, x2, y, cv=5, scoring='neg_mean_squared_error')

# Print the average score and its standard deviation
print("Cross-validation scores:", scores)
print("Average score:", scores.mean())
print("Standard deviation:", scores.std())

nums = list(range(0,len(y_test2)))

plt.plot(nums, y_test2, color="red")
plt.plot(nums, y_pred2, color="yellow")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print("---")


print(X_test3[:2])
print(regr3.predict(X_test3[:1]))

print(regr3.score(X_test3, y_test3))

y_pred3 = regr3.predict(X_test3)
acc = metrics.r2_score(y_test3, y_pred3)
print(acc)

scores = cross_val_score(regr3, x3, y, cv=5, scoring='neg_mean_squared_error')

# Print the average score and its standard deviation
print("Cross-validation scores:", scores)
print("Average score:", scores.mean())
print("Standard deviation:", scores.std())

nums = list(range(0,len(y_test3)))

plt.plot(nums, y_test3, color="red")
plt.plot(nums, y_pred3, color="yellow")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print("-----")

print(f"types - {len(X_test3)} and {len(y_test3)}")

X_test3 = np.array(X_test3)
X_test3 = X_test3.reshape(-1, 3)
y_test3 = np.array(y_test3)

results = model.evaluate(np.array(X_test3), np.array(y_test3))
print(f'Test loss: {results:.3f}')

# Get predictions for the test data
y_pred = model.predict(np.array(X_test3))

# Calculate mean squared error
mse = mean_squared_error(y_test3, y_pred)
print(f"Mean squared error: {mse:.3f}")

# Calculate R-squared score
r2 = r2_score(y_test3, y_pred)
print(f"R-squared score: {r2:.3f}")

print(model.predict(np.array(X_test3[:1])))


nums = list(range(0,len(y_test3)))

plt.plot(nums, y_test3, color="red")
plt.plot(nums, y_pred, color="yellow")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print("-----")

print(f"types - {len(X_test2)} and {len(y_test2)}")

X_test2 = np.array(X_test2)
X_test2 = X_test2.reshape(-1, 3)
y_test2 = np.array(y_test2)

results = model2.evaluate(np.array(X_test2), np.array(y_test2))
print(f'Test loss: {results:.3f}')

# Get predictions for the test data
y_pred2 = model2.predict(np.array(X_test2))

# Calculate mean squared error
mse = mean_squared_error(y_test2, y_pred2)
print(f"Mean squared error: {mse:.3f}")

# Calculate R-squared score
r2 = r2_score(y_test2, y_pred2)
print(f"R-squared score: {r2:.3f}")

print(model.predict(np.array(X_test2[:1])))

nums = list(range(0,len(y_test2)))

plt.plot(nums, y_test2, color="red")
plt.plot(nums, y_pred2, color="yellow")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print("-----")

print(f"types - {len(X_test)} and {len(y_test)}")

X_test = np.array(X_test)
y_test = np.array(y_test)

print(len(X_test), len(y_test))

results = model3.evaluate(np.array(X_test), np.array(y_test))
print(f'Test loss: {results:.3f}')

# Get predictions for the test data
y_pred3 = model3.predict(np.array(X_test))

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred3)
print(f"Mean squared error: {mse:.3f}")

# Calculate R-squared score
r2 = r2_score(y_test, y_pred3)
print(f"R-squared score: {r2:.3f}")

print(model3.predict(np.array(X_test[:1])))


nums = list(range(0,len(y_test)))

plt.plot(nums, y_test, color="red")
plt.plot(nums, y_pred3, color="yellow")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print(f"Total inputs: {stuffie}")


def determine_value(value):
    global model

    value = np.array(value).reshape(1, -1)

    print(f"value: {type(value)} and {type(X_test3)}")


    result = model.predict(value)
    print(result)

    if result[0][0] < 0:
        result[0][0] = result[0][0] * -1
    
    publish(client, result)


client = connect_mqtt()
client.loop_start()


