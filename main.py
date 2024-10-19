# in this part of the project we use the libraries like numpy and pandas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# in this part we import the operating system too
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# and also we read the the data collection
data=pd.read_csv('/kaggle/input/spam-email/spam.csv')
data
# we inlcude the columns
data.columns
# and also shows the info of the data
data.info()

data.isna().sum()
# in this part we use use the spam identification 
data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)

#CounterVectorizer Convert the text into matrics
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

clf.fit(X_train,y_train)

emails=[
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]

clf.predict(emails)

clf.score(X_test,y_test)



