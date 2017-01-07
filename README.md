# question_classification
A basic text features based classifier that can predict the type of question.

Dependency 

* scikit-learn
* numpy
* scipy

I would recommand downloading anaconda python https://www.continuum.io/downloads But you can also install individual libs.

```
pip install sklearn

```
How to run 

1 Training 

```
python question_classification.py 1 LabelledData\ \(1\).txt 
```
2  Testing 

```
python question_classification.py 2 train_1000.label.txt model.pkl vect.pkl
```
