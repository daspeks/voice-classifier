[Test](#Test)




```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
df = pd.read_csv('voice.csv', header=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meanfreq</th>
      <th>sd</th>
      <th>median</th>
      <th>Q25</th>
      <th>Q75</th>
      <th>IQR</th>
      <th>skew</th>
      <th>kurt</th>
      <th>sp.ent</th>
      <th>sfm</th>
      <th>...</th>
      <th>centroid</th>
      <th>meanfun</th>
      <th>minfun</th>
      <th>maxfun</th>
      <th>meandom</th>
      <th>mindom</th>
      <th>maxdom</th>
      <th>dfrange</th>
      <th>modindx</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.059781</td>
      <td>0.064241</td>
      <td>0.032027</td>
      <td>0.015071</td>
      <td>0.090193</td>
      <td>0.075122</td>
      <td>12.863462</td>
      <td>274.402906</td>
      <td>0.893369</td>
      <td>0.491918</td>
      <td>...</td>
      <td>0.059781</td>
      <td>0.084279</td>
      <td>0.015702</td>
      <td>0.275862</td>
      <td>0.007812</td>
      <td>0.007812</td>
      <td>0.007812</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>male</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.066009</td>
      <td>0.067310</td>
      <td>0.040229</td>
      <td>0.019414</td>
      <td>0.092666</td>
      <td>0.073252</td>
      <td>22.423285</td>
      <td>634.613855</td>
      <td>0.892193</td>
      <td>0.513724</td>
      <td>...</td>
      <td>0.066009</td>
      <td>0.107937</td>
      <td>0.015826</td>
      <td>0.250000</td>
      <td>0.009014</td>
      <td>0.007812</td>
      <td>0.054688</td>
      <td>0.046875</td>
      <td>0.052632</td>
      <td>male</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.077316</td>
      <td>0.083829</td>
      <td>0.036718</td>
      <td>0.008701</td>
      <td>0.131908</td>
      <td>0.123207</td>
      <td>30.757155</td>
      <td>1024.927705</td>
      <td>0.846389</td>
      <td>0.478905</td>
      <td>...</td>
      <td>0.077316</td>
      <td>0.098706</td>
      <td>0.015656</td>
      <td>0.271186</td>
      <td>0.007990</td>
      <td>0.007812</td>
      <td>0.015625</td>
      <td>0.007812</td>
      <td>0.046512</td>
      <td>male</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.151228</td>
      <td>0.072111</td>
      <td>0.158011</td>
      <td>0.096582</td>
      <td>0.207955</td>
      <td>0.111374</td>
      <td>1.232831</td>
      <td>4.177296</td>
      <td>0.963322</td>
      <td>0.727232</td>
      <td>...</td>
      <td>0.151228</td>
      <td>0.088965</td>
      <td>0.017798</td>
      <td>0.250000</td>
      <td>0.201497</td>
      <td>0.007812</td>
      <td>0.562500</td>
      <td>0.554688</td>
      <td>0.247119</td>
      <td>male</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.135120</td>
      <td>0.079146</td>
      <td>0.124656</td>
      <td>0.078720</td>
      <td>0.206045</td>
      <td>0.127325</td>
      <td>1.101174</td>
      <td>4.333713</td>
      <td>0.971955</td>
      <td>0.783568</td>
      <td>...</td>
      <td>0.135120</td>
      <td>0.106398</td>
      <td>0.016931</td>
      <td>0.266667</td>
      <td>0.712812</td>
      <td>0.007812</td>
      <td>5.484375</td>
      <td>5.476562</td>
      <td>0.208274</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# check for null values
df.isna().sum()
```




    meanfreq    0
    sd          0
    median      0
    Q25         0
    Q75         0
    IQR         0
    skew        0
    kurt        0
    sp.ent      0
    sfm         0
    mode        0
    centroid    0
    meanfun     0
    minfun      0
    maxfun      0
    meandom     0
    mindom      0
    maxdom      0
    dfrange     0
    modindx     0
    label       0
    dtype: int64



**Note:** 
* To detect NaN values numpy uses `np.isnan()`.
* To detect NaN values pandas uses either `.isna()` or `.isnull()`.
* The NaN values are inherited from the fact that pandas is built on top of numpy, while the two functions' names originate from R's DataFrames, whose structure and functionality pandas tried to mimic.


```python
# check shape
df.shape
```




    (3168, 21)




```python
# print some label metadata
print ("Total number of labels: {}".format(df.shape[0]))
print ("Number of male labels: {}".format(df[df['label']=='male'].shape[0]))
print ("Number of female labels: {}".format(df[df['label']=='female'].shape[0]))
```

    Total number of labels: 3168
    Number of male labels: 1584
    Number of female labels: 1584



```python
# Scale the features
from sklearn.preprocessing import StandardScaler

X = df.iloc[:, :-1]
X = StandardScaler().fit(X).transform(X)
```


```python
# Encode labels to int
from sklearn.preprocessing import LabelEncoder
df['label'] = LabelEncoder().fit_transform(df['label'])

# Set y to labels
y = df.iloc[:, -1]
```


```python
# Split the dataset into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```


```python
# Plot a correlation matrix
f = plt.figure(figsize=(20, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
```


![png](output_11_0.png)



```python
# Run SVM with default hyperparameter
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score

svc = SVC() # default hyperparameters
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print ('Accuracy score: {:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))
print ('Mean cross validation score: {:.4f}'.format(scores.mean()))
```

    Accuracy score: 0.9763
    Mean cross validation score: 0.9665



```python
# Run SVM with linear kernel
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print ('Accuracy score: {:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))
print ('Mean cross validation score: {:.4f}'.format(scores.mean()))
```

    Accuracy score: 0.9779
    Mean cross validation score: 0.9697



```python
# Run SVM with RBF (Radial Basis Function) kernel
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print ('Accuracy score: {:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))
print ('Mean cross validation score: {:.4f}'.format(scores.mean()))
```

    Accuracy score: 0.9763
    Mean cross validation score: 0.9665



```python
# Run SVM with polynomial kernel

svc = SVC(kernel='poly', degree=3)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print ('Accuracy score: {:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))
print ('Mean cross validation score: {:.4f}'.format(scores.mean()))
```

    Accuracy score: 0.9590
    Mean cross validation score: 0.9451


**Note:** Polynomial kernel's poor performance compared to the other kernels might be due to the fact that it is overfitting the training set.

Through cross validation scores, we see how the accuracy score differs with each random splitting of data. In other words, we see that the accuracy score is a variant under splitting of the dataset. 

In K-fold cross validation, we usually take the mean of all scores.


```python
# Toying with the C hyperparameter using a linear kernel
# Using a coarse grain of C values
Clist = list (range(1,26))
acc_score = []
for c in Clist:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
```


```python
plt.plot(Clist, acc_score)
plt.xticks(np.arange(0,27,2))
plt.xlabel('C value')
plt.ylabel('K-fold cross validation accuracy score')
plt.show();
```


![png](output_18_0.png)



```python
# Using a fine grain of C values
Clist = list (np.arange(0.1,6,0.1))
acc_score = []
for c in Clist:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
```


```python
plt.figure(figsize=(10, 10))
plt.plot(Clist, acc_score)
plt.xticks(np.arange(0.0,6,0.3))
plt.xlabel('C value')
plt.ylabel('K-fold cross validation accuracy score')
plt.show();
```


![png](output_20_0.png)


Thus, using a finer grain for the C hyperparameter, we see that for C = 0.1, we get the highest accuracy score.


```python
# Toying with the gamma and C hyperparameter using a RBF kernel
# coarse grain
gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
acc_score = []
for g in gamma_list:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())

plt.figure(figsize=(10, 10))
plt.plot(gamma_list, acc_score)
plt.xticks(np.arange(0.0001,100,5))
plt.xlabel('Gamma value')
plt.ylabel('K-fold cross validation accuracy score')
plt.show();
```


![png](output_22_0.png)


The gamma hyperparameter is the inverse of the standard deviation in the RBF kernel. Thus, increasing gamma results in decreasing standard deviation and the boundary function ends up being more irregular and each instance’s range of influence is smaller. Conversely, a small gamma value results in a larger standard deviation, so instances have a larger range of influence and the decision boundary ends up smoother. 

Thus, gamma acts like a regularization hyperparameter:
* if your model is overfitting, reduce gamma
* if your model is underfitting, increase gamma

similar to the C hyperparameter


```python
# Toying with the gamma and C hyperparameter using a RBF kernel
# fine grain
gamma_list = [0.0001, 0.001, 0.01, 0.1]
acc_score = []
for g in gamma_list:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())

plt.figure(figsize=(10, 10))
plt.plot(gamma_list, acc_score)
plt.xticks(np.arange(0.0001,0.1,0.01))
plt.xlabel('Gamma value')
plt.ylabel('K-fold cross validation accuracy score')
plt.show();
```


![png](output_24_0.png)


Thus, using a finer grain for the gamma hyperparameter, we see that for gamma value ~ 0.01, we get the highest accuracy score.


```python
# Toying with the degree hyperparameter and using a polynomial kernel
# coarse grain
degree_list = [2,3,4,5,6]
acc_score = []
for d in degree_list:
    svc = SVC(kernel='poly', degree=d)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())

plt.figure(figsize=(10, 10))
plt.plot(degree_list, acc_score)
plt.xlabel('Degree of polynomial')
plt.ylabel('K-fold cross validation accuracy score')
plt.show();
```


![png](output_26_0.png)


We see that the accuracy score is maximum with degree 3. Higher degrees perform poorly as they overfit the data.


```python
# Now run SVM with new found optimal parameters

# Linear
svc = SVC(kernel='linear', C=0.1)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
accuracy_score = metrics.accuracy_score(y_test, y_predict)

# Linear with K-fold
svc = SVC(kernel='linear', C=0.1)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print ('Linear kernel accuracy: {:.2f}%'.format(accuracy_score*100))
print ('Linear kernel + K-fold accuracy: {:.2f}%'.format(scores.mean()*100))
```

    Linear kernel accuracy: 97.48%
    Linear kernel + K-fold accuracy: 97.06%


Even though the accuracy of the model is slightly better without using K-fold validation, the model may fail to properly generalise on unseen data.

Thus, it is usually a good idea to perform K-fold cross validation.


```python
# RBF
svc = SVC(kernel='rbf', gamma=0.01)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
accuracy_score = metrics.accuracy_score(y_test, y_predict)

# RBF with K-fold
svc = SVC(kernel='rbf', gamma=0.1)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print ('Linear kernel accuracy: {:.2f}%'.format(accuracy_score*100))
print ('Linear kernel + K-fold accuracy: {:.2f}%'.format(scores.mean()*100))
```

    Linear kernel accuracy: 96.69%
    Linear kernel + K-fold accuracy: 96.37%



```python
# Polynomial
svc = SVC(kernel='poly', degree=3)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
accuracy_score = metrics.accuracy_score(y_test, y_predict)

# Linear with K-fold
svc = SVC(kernel='poly', degree=3)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print ('Linear kernel accuracy: {:.2f}%'.format(accuracy_score*100))
print ('Linear kernel + K-fold accuracy: {:.2f}%'.format(scores.mean()*100))
```

    Linear kernel accuracy: 95.90%
    Linear kernel + K-fold accuracy: 94.51%



```python
# Use grid search method to find the best parameters

tuned_params = {
    'C'     : (np.arange(0.1,1.1,0.1)),
    'kernel': ['linear'],
    
    'C'     : (np.arange(0.1,1.1,0.1)),
    'gamma' : (np.arange(0.01,0.06,0.01)),
    'kernel': ['rbf'],
    
    'C'     : (np.arange(0.1,1.1,0.1)),
    'gamma' : (np.arange(0.01,0.06,0.01)),
    'degree': (np.arange(2,5,1)),
    'kernel': ['poly']
}

tuned_param = {
    'C'     : (np.arange(0.0,1.1,0.1)),
    'gamma' : (np.arange(0.00,0.06,0.01)),
    'degree': (np.arange(0,5,1)),
    'kernel': ['linear', 'rbf', 'poly'],
}
```


```python
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.model_selection import GridSearchCV

model = GridSearchCV (svc, tuned_param, cv=10, scoring='accuracy')
```


```python
model.fit(X_train, y_train)
print (model.best_score_)
```

    0.9814556036226696


# Test


```python
print (model.best_params_)
```

    {'C': 0.9, 'degree': 0, 'gamma': 0.05, 'kernel': 'rbf'}



```python
y_pred = model.predict(X_test)
print ('Accuracy: {:.2f}%'.format(metrics.accuracy_score(y_pred, y_test)))
```

    Accuracy: 0.98%



```python

```
