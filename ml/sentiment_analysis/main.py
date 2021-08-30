import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

"""Reading the dataset"""
df_review = pd.read_csv('IMDB Dataset.csv')
df_positive = df_review[df_review['sentiment'] == 'positive'][:9000]
df_negative = df_review[df_review['sentiment'] == 'negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative])

"""Dealing with Imbalanced Classes"""
rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(df_review_imb[['review']], df_review_imb['sentiment'])

"""Splitting data into train and test set"""
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

"""Turning our text data into numerical vectors"""
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

"""Different models"""
"""Support Vector Machines(SVM)"""
svc = SVC(C=1, kernel='linear')
svc.fit(train_x_vector, train_y)

"""Decision Tree"""
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

"""Naive Bayes"""
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

"""Logistic Regression"""
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

"""Model Evaluation"""
"""Mean Accuracy"""
print("SVC score: ", svc.score(test_x_vector, test_y))
print("Decision tree score: ", dec_tree.score(test_x_vector, test_y))
print("Naive Bayes score: ", gnb.score(test_x_vector.toarray(), test_y))
print("Logistic Regression score: ", log_reg.score(test_x_vector, test_y))

"""F1 score"""
print(f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None))

"""Classification report"""
print(classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative']))

"""Confusion Matrix"""
conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])
print(conf_mat)

"""Tuning the Model"""
"""GridSearchCV"""
# parameters = {'C': [1, 4, 8, 16, 32], 'kernel': ['linear', 'rbf']}
# svc = SVC()
# svc_grid = GridSearchCV(svc, parameters, cv=5)
# svc_grid.fit(train_x_vector, train_y)
#
# print(svc_grid.best_params_) # Output: "{'C': 1, 'kernel': 'linear'}"
# print(svc_grid.best_estimator_) # Output: "SVC(C=1, kernel='linear')"

"""User Interface"""
while True:
    inp = input('Write review, or "q" for quit the program: ')
    if inp == 'q':
        exit()
    else:
        print(svc.predict(tfidf.transform([inp])))
