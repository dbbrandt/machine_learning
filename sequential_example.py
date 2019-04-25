
# example making new class prediction for a classification problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

from numpy import array
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
print("Type of y: {}".format(type(y)))
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

early_stopping_monitor = EarlyStopping(patience=3)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=1,  callbacks=[early_stopping_monitor])
# new instance where we do not know the answer
# Xnew = array([[0.89337759, 0.65864154]])
Xnew = array([[0.21337759, 0.12864154]])
# make a prediction
ynew = model.predict_classes(Xnew)
# show the inputs and predicted outputs

print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
