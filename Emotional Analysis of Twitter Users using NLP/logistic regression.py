import pandas as pd;
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def Data_preprocessing(df):
    col = ['Sentiment', 'SentimentText'];
    df = df[col];   # Get the cells whose column heading is 'col'
    df = df[pd.notnull(df['SentimentText'])];
    df.columns = ['Sentiment', 'SentimentText'];
    df['category_id'] = df['Sentiment'].factorize()[0];
    category_id_df = df[['Sentiment', 'category_id']].drop_duplicates().sort_values('category_id');
    category_to_id = dict(category_id_df.values);   # Creating dictionaries for future use
    id_to_category = dict(category_id_df[['category_id', 'Sentiment']].values);
    fig = plt.figure(figsize=(8,6));
    df.groupby('Sentiment').SentimentText.count().plot.bar(ylim=0);
    plt.show(); # shows what percent of output is '1' or '0'
    return df, category_id_df

def Vectorize(data):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df = 10, norm='l2', encoding='mac_roman', ngram_range=(1, 1), stop_words='english');
    features = tfidf.fit_transform(data.SentimentText).toarray(); # calculate a tf-idf vector for each tweet
    labels = data.category_id;
    count_vect = CountVectorizer();
    X_train_counts = count_vect.fit_transform(data.SentimentText); # Transforming the tweet to numeric array
    tfidf_transformer = TfidfTransformer();
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts); # Weighing the tweet based on TF-IDF(Term Frequency - Inverse Document Frequency)
    return X_train_tfidf, features, labels;

def Model_Selection(features, labels):
    models = [RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), LinearSVC(), MultinomialNB(), LogisticRegression(random_state=0),]; #Trying all the ML models to find the model that gives best accuracy.
    CV = 5;
    cv_df = pd.DataFrame(index=range(CV * len(models)));
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV); # Calculating accuracies for various models
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy));
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy']);
    sns.boxplot(x='model_name', y='accuracy', data=cv_df);
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2);
    plt.show(); # plots the accuracy of all the models.
    print(cv_df.groupby('model_name').accuracy.mean());
    return

def Model_Training(data, features, labels):
    model = LogisticRegression(random_state=0); # Logistic regression model gave best accuracy
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.33, random_state=0);
    clf = model.fit(X_train, y_train); # Training the Logistic regression model
    return clf, X_test, y_test

def Model_Testing(data, model, category_id_data, X_test, y_test):
    y_pred = model.predict(X_test); # Evaluating the model using test data
    conf_mat = confusion_matrix(y_test, y_pred); # Calculating the confusion matrix
    fig, ax = plt.subplots(figsize=(10,10));
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_data.Sentiment.values, yticklabels=category_id_data.Sentiment.values);
    plt.ylabel('Actual');
    plt.xlabel('Predicted');
    plt.show(); # plots the confusion matrix
    #print(metrics.classification_report(y_test, y_pred, target_names=data['Sentiment'].unique())); # prints the classification metrics
    return

def logreg(loc):
    df = pd.read_csv(loc, encoding='mac_roman'); #Read the .csv file that contains the data
    [data, category_id_data] = Data_preprocessing(df);
    [X_train_tfidf, features, labels] = Vectorize(data);
    Model_Selection(features, labels);
    [clf, X_test, y_test] = Model_Training(data, features, labels);
    Model_Testing(data, clf, category_id_data, X_test, y_test);
    return clf

def predict_tweets(clf, loc):
    tweets = pd.read_csv(loc);
    [data, category_id_data] = Data_preprocessing(tweets);
    [X_train_tfidf, features, labels] = Vectorize(data);
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0, random_state=0);
    print(clf.predict(X_train));
    return

clf = logreg('C:/Users/akhil/Desktop/Emotional Analysis of Twitter Users using NLP/Data.csv'); # path to Data file

'''
N = 2
for Sentiment, category_id in sorted(category_to_id.items()):  # Finding the terms that are the most correlated with output values '0' and '1'
	  features_chi2 = chi2(features, labels == category_id)
	  indices = np.argsort(features_chi2[0])
	  feature_names = np.array(tfidf.get_feature_names())[indices]
	  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
	  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
	  print("# '{}':".format(Sentiment))
	  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
	  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
'''

#X_train, X_test, y_train, y_test = train_test_split(df['SentimentText'], df['Sentiment'], random_state = 0); # Splitting data into training data and testing data.



#clf = MultinomialNB().fit(X_train_tfidf, y_train); # Fitting the Data to Multinomial Naive Bayes classifier



#predict_tweets(clf, 'D:/Masters/Data Mining/Twitter Sentimental Analysis/Data.csv');



