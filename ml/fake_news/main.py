import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data_head = 'text'
label_head = 'label'
label_0 = 'FAKE'
label_1 = 'REAL'

"""Reading the dataset"""
df_news = pd.read_csv('news.csv')

"""Splitting data into train and test set"""
train, test = train_test_split(df_news, test_size=0.1, random_state=42)
train_x, train_y = train[data_head], train[label_head]
test_x, test_y = test[data_head], test[label_head]

"""Turning our text data into numerical vectors"""
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

"""PassiveAggressiveClassifier"""
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(train_x_vector, train_y)

"""Mean Accuracy"""
print(pac.score(test_x_vector, test_y))

"""F1 score"""
print(f1_score(test_y, pac.predict(test_x_vector), labels=[label_1, label_0], average=None))

"""Classification report"""
print(classification_report(test_y, pac.predict(test_x_vector), labels=[label_1, label_0]))

"""Confusion Matrix"""
conf_mat = confusion_matrix(test_y, pac.predict(test_x_vector), labels=[label_1, label_0])
print(conf_mat)

"""User Interface"""
while True:
    inp = input('Write review, or "q" for quit the program: ')
    if inp == 'q':
        exit()
    else:
        print(pac.predict(tfidf.transform([inp])))
