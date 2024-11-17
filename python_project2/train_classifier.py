import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_ = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_['data'])
output = np.asarray(data_['labels'])


x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=0.2, shuffle=True, stratTrueify=output)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()