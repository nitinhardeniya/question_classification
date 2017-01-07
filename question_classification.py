#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
import time as t 
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report




def readdata(datafile):
    y=[]
    X=[]
    f=open(datafile,'r')
    for line in f:
        parts=line.strip().split(',,,')
        question_class=parts[-1]
        question=parts[0]
        X.append(question)
        y.append(question_class)
    return X,y



def preprocess(X,y):
	""" 
	    this function takes a pre-made list of questions 
	    and the corresponding question_class
	    a number of preprocessing steps:
	        -- splits into training/testing sets (10% testing)
	        -- vectorizes into tfidf matrix
	        -- selects/keeps most helpful features

	    after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

	    4 objects are returned:
	        -- training/testing features
	        -- training/testing labels

	"""
	### text vectorization--go from strings to lists of numbers
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	print ("dimention of the traing data "+ str(len(X_train)))
	print ("dimention of the test data "+ str(len(X_test)))
	vectorizer = CountVectorizer(min_df=10)
	#vectorizer=TfidfVectorizer()

	# lot more can be done as POS based feature or NER based feature
	# We just need tostack them and then again the feature vectors should look the same

	features_train_transformed = vectorizer.fit_transform(X_train)
	features_test_transformed  = vectorizer.transform(X_test)

	return features_train_transformed, features_test_transformed, y_train, y_test,vectorizer



def model_training(corpus_file):
	X,y=readdata(corpus_file)
	features_train_transformed, features_test_transformed, y_train, y_test,vectorizer=preprocess(X,y)
	t0 = t.time()
	#clf = svm.SVC(kernel='rbf',C=10000)
	#clf=SGDClassifier(loss="hinge", penalty="l2")
	clf=SGDClassifier(loss="hinge", penalty="elasticnet",l1_ratio=0.25)

	#clf = LogisticRegression()

	clf.fit(features_train_transformed,y_train)
	print "training time:", round(t.time()-t0, 3), "s"
	t0 = t.time()
	pred_test=clf.predict(features_test_transformed)
	question_types=["affirmation","what","who","unknow","when"]
	print(classification_report(y_test, pred_test))
	import pickle
	filename = 'model.pkl'
	vectorizer_file='vect.pkl'
	pickle.dump(clf, open(filename, 'wb'))
	pickle.dump(vectorizer, open(vectorizer_file, 'wb'))
	return clf,pred_test

def model_test(testfile,modelpkl,vectpkl):
	f=open(testfile,'r')
	model = pickle.load(open(modelpkl, 'rb'))
	vectorizer=pickle.load(open(vectpkl, 'rb'))
	for line in f:
	    try:
	        parts=line.strip().split(':')
	        #print parts
	        question_class=parts[0]
	        question=str(parts[1]).decode("utf-8")
	        test_question=vectorizer.transform([question])
	        #print test_question

	        print question_class +"\t"+ question +"\t"+ str(model.predict(test_question)) 
	    except:
	        "Something with the input"

def main():
	if sys.argv<2:
		print "Please provide more arguments "
		print "for training use <question_classification.py><1><train file>"
		print "for testing use <question_classification.py><2><test file><model.pkl><vect.pkl>"
	elif sys.argv[1]=="1":
		model_training(sys.argv[2])
	elif sys.argv[1]=="2":
		model_test(sys.argv[2],sys.argv[3],sys.argv[4])

if __name__ == '__main__':
	main()
