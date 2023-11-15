# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
## Program:
```py
## Program to implement the SVM For Spam Mail Detection..
## Developed by: Kavinraja D
## RegisterNumber:  212222240047

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### Result output

![ml901](https://github.com/A-Thiyagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707693/1be2e57f-2501-41c0-862a-19bd02626dc6)
###  data.head()

![ml902](https://github.com/A-Thiyagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707693/afdcd24d-f5f0-48e8-ac1f-6fa351dc640d)
###  data.info()


![ml903](https://github.com/A-Thiyagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707693/56281d06-3be6-42b9-b41c-6022904ee09f)
### data.isnull().sum()

![ml904](https://github.com/A-Thiyagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707693/d5bb37c1-052e-46c8-b99f-ef52ad4996bb)
### Y_prediction value

![ml905](https://github.com/A-Thiyagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707693/c709b158-e17a-497d-923b-122cff2eff12)
###  Accuracy value
![ml906](https://github.com/A-Thiyagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707693/d2dbf4c8-9e19-4d3a-ab23-ecb68d490c99)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
