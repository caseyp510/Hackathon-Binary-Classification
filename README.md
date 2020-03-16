
---
### Problem Statement

Data Science supervised binary classification project using `limited data` within  `7 hours` as constraints 

---
### Goal

To find the best possible way to predict  if wages >= 50k using only 6513 observations and 14 features as best as possible. 
The actual predictions for 16,281 observations and 14 features.

---

### Solution Found

This project was a one day group project that started at 9 am and ended at 4 pm. My contribution to it was to create a python script framework that would automatically select features and run through different classifiers like 
LogisticRegression,  MultinomialNB, KNeighborsClassifier, GaussianNB, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier using 
GridSearch/GridSearchCV with multiple paramters and PolynomialFeatures

The best classifier : GradientBoostingClassifier(max_depth=3, n_estimators=100,
learning_rate=0.1) with 0.83% accuracy.

    Features : Age, hours per week, marital-status, education num, sex , workclass, country
    Bin Fields - Age, hours per week
    Dummy fields - marital-status,sex , workclass and country
    Ignored Fields: fnlwgt, education, capital-gain, capital-loss
    Predicting the test data (16,281) for submission:
        13111 <= 50k
        3170 >50k

---
Our accuracy on the test data upon submission was : 96% for the test data

___

Link to Python script used : (./python_codes/classification_framework.py)




