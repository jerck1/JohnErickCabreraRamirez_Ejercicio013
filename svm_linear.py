#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[2]:


def prepare_data():
    # lee los numeros
    numeros = skdata.load_digits()

    # lee los labels
    target = numeros['target']

    # lee las imagenes
    imagenes = numeros['images']

    # cuenta el numero de imagenes total
    n_imagenes = len(target)

    # para poder correr PCA debemos "aplanar las imagenes"
    data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

    # Split en train/test
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8)

    # todo lo que es diferente de 1 queda marcado como 0
    y_train[y_train!=1]=0
    y_test[y_test!=1]=0

    # Reescalado de los datos
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return {'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test}


# In[3]:


data=prepare_data()
cat=['x_train', 'x_test', 'y_train', 'y_test']
#data


# In[4]:


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(kernel='linear',gamma='auto')
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))
np.shape(data['x_train'])


# In[5]:


clf = SVC(C=10000000000,kernel='linear',gamma='auto')
#print(clf.fit(data['x_train'], data['y_train']))
print(clf)
clf.fit(data['x_train'], data['y_train'])
predicted=clf.predict(data['x_train'])


# In[6]:


for i in range(1,10):
    clf = SVC(C=i,kernel='linear',gamma='auto')
#print(clf.fit(data['x_train'], data['y_train']))
    print(clf)
    clf.fit(data['x_train'], data['y_train'])
    predicted=clf.predict(data['x_train'])
    print(predicted)
    #if(predicted<)


# In[7]:


params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
          'gamma': [0.0001, 0.001, 0.01, 0.1],
          'kernel':['linear','rbf'] }


# In[8]:


#grid_clf = GridSearchCV(SVC(class_weight='balanced'), params_grid)


# In[9]:


#grid_clf = clf.fit(data['x_train'], data['y_train'])


# In[10]:


#print(grid_clf.best_estimators)


# In[ ]:




