from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.base import TransformerMixin
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd


from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


rs = 42
logreg_penalty = ['l1', 'l2']
logreg_max_iter = [100]
knn_neighbors = [5, 10]
knn_p = [1, 2]
dt_max_depth = [3]
gb_max_depth = [3]
gb_learning_rate = [0.1]
gb_n_estimators = [100]

classification_params = {
    1: {"Pipeline": [('classifier', LogisticRegression(random_state=rs))],
        "hyper_params": {
            'classifier__penalty': logreg_penalty,
            'classifier__max_iter': logreg_max_iter}
        },
    2: {"Pipeline": [('classifier', LogisticRegressionCV(random_state=rs))],
        "hyper_params": {}
        },
    3: {"Pipeline": [('classifier', MultinomialNB())],
        "hyper_params": {},
        },
    4: {"Pipeline": [('ss', StandardScaler()),
                     ('classifier', KNeighborsClassifier())],
        "hyper_params": {
            'classifier__n_neighbors': knn_neighbors,
            'classifier__p': knn_p}
        },
    5: {"Pipeline": [('classifier', KNeighborsClassifier())],
        "hyper_params": {
            'classifier__n_neighbors': knn_neighbors,
            'classifier__p': knn_p}
        },
    6: {"Pipeline": [('classifier', GaussianNB())],
        "hyper_params": {}
        },
    7: {"Pipeline": [('classifier', DecisionTreeClassifier(random_state=42))],
        "hyper_params": {'classifier__max_depth': dt_max_depth}
        },
    8: {"Pipeline": [('classifier', RandomForestClassifier(random_state=42))],
        "hyper_params": {}
        },
    9: {"Pipeline": [('classifier', ExtraTreesClassifier(random_state=42))],
        "hyper_params": {}
        },
    10: {"Pipeline": [('classifier', GradientBoostingClassifier(random_state=rs))],
        "hyper_params": {
            'classifier__max_depth': gb_max_depth,
            'classifier__learning_rate': gb_learning_rate,
            'classifier__n_estimators': gb_n_estimators}
         },
    11: {"Pipeline": [('classifier', SVC())],
         "hyper_params": {}
         },
    12: {"Pipeline": [('classifier', LinearSVC())],
         "hyper_params": {}
         }
    # 12: {"Pipeline": [('classifier', NuSVC())],
    #      "hyper_params": {}
    #      }
}


def get_boosted_scores(estimator,data, y_col, final_feature_set):
    """

    :param estimator:
    :return:
    """
    boosters = {
        1: {"Pipeline": [('classifier', BaggingClassifier(random_state=rs))],
            "hyper_params": {
                'classifier__base_estimator': [estimator] }
            },
        2: {"Pipeline": [('classifier', AdaBoostClassifier(random_state=rs))],
            "hyper_params": {
                'classifier__base_estimator': [estimator] }
            },
        }

    params = {
        1: {"Pipeline": [('classifier', estimator)],
            "hyper_params": {}
            },
        2: {"Pipeline": [('classifier', estimator)],
            "hyper_params": {}
           },
        }

    X = data.drop(columns=[y_col])
    y = data[y_col]
    feature_sets = get_feature_sets(X, y, final_feature_set,params)
    best_scores = {x: {y: [] for y in boosters} for x in range(1, 3)}

    for bs in boosters:
        for feature_no in range(0, 2):
            features = feature_sets[bs][feature_no]
            if type(features) is list:
                best_scores[feature_no + 1][bs] = model_fit_score(data[features], data[y_col], feature_no + 1, bs, boosters)

    return best_scores


# return best estimator results
def get_score_data(results, X_train, X_test, y_train, y_test):
    """

    :param results:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    output = []
    sub_scores = {}

   #choosing the best estimator of the lot
    model = results.best_estimator_

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()

    ##calulating accuracy
    #     accuracy = (tp + tn) / (tp + fp + tn + fn)
    accuracy = round(accuracy_score(y_test, model.predict(X_test)), 2)

    # calculating Misclassification Rate
    mis_calcuations = 1 - accuracy

    # calculating sensitivity
    sensitivity = tp / (tp + fn)

    # calculating specificity
    specificity = tn / (tn + fp)

    # calculating precision
    precision = tp / (tp + fp)

    # to predict roc_auc_score
    try:
        pred_proba = [i[1] for i in model.predict_proba(X_test)]
    except:
        pred_proba = [0 for i in y_test]

    pred_df = pd.DataFrame({'true_values': y_test,
                            'pred_probs': pred_proba})



    # For returning results from the best estimator
    # 1. best score
    output.append(round(results.best_score_, 2))

    # 2.best params
    output.append(results.best_params_)

    # 3.No of 0's and 1's in test
    sub_scores.update({"Not eligible for 401k": y_test[y_test == 0].count()})
    sub_scores.update({"Eligible for 401k": y_test[y_test == 1].count()})

    # 4.No of 0's and 1's predicted
    df = pd.DataFrame({"Preds": model.predict(X_test)})
    try:
        sub_scores.update({"Not eligible for 401k": df.groupby("Preds")["Preds"].value_counts()[0].values[0]})
    except:
        sub_scores.update({"Not eligible for 401k":0})

    try:
        sub_scores.update({"Eligible for 401k": df.groupby("Preds")["Preds"].value_counts()[1].values[0]})
    except:
        sub_scores.update({"Eligible for 401k": 0})

    # 5.baseline
    sub_scores.update({"Baseline accuracy%": round(y_test.value_counts(normalize=True)[0], 2)})

    # 6.Train Score
    sub_scores.update({"Train Scores": round(model.score(X_train, y_train), 2)})

    # 7.Test Score
    sub_scores.update({"Test Scores": round(model.score(X_test, y_test), 2)})

    # 8.Accuracy
    sub_scores.update({"Accuracy": accuracy})

    # 9.Mis Calculations
    sub_scores.update({"Mis Calculations": round(mis_calcuations, 2)})

    # 10.Sensitivity
    sub_scores.update({"Sensitivity": round(sensitivity, 2)})

    # 11.Specificity
    sub_scores.update({"Specificity": round(specificity, 2)})

    # 12.Precision
    sub_scores.update({"Precision": round(precision, 2)})

    # 13.ROC AUC
    sub_scores.update({"ROC AUC": round(roc_auc_score(pred_df['true_values'], pred_df['pred_probs']), 2)})
    #
    # # 14.ROC AUC curve
    # sub_scores.update({"ROC AUC curve": round(roc_curve(pred_df['true_values'], pred_df['pred_probs']), 2)})

    # 15.F1 Train Score
    sub_scores.update({"F1 score Train": round(f1_score(y_train, model.predict(X_train)), 2)})

    # 16.F1 Test Score
    sub_scores.update({"F1 score Test": round(f1_score(y_test, model.predict(X_test)), 2)})

    # 17.Log loss score
    sub_scores.update({"Log Loss score": round(log_loss(y_test, model.predict(X_test)), 2)})

    # 18.
    if model.score(X_train, y_train) > model.score(X_test, y_test):
        sub_scores.update({"Fit Type": "Overfit"})
    else:
        sub_scores.update({"Fit Type": "Underfit"})

    output.append(sub_scores)

    return output


def model_fit_score(X, y, feature_no, cp, params):
    """

    :param X:
    :param y:
    :param best_scores:
    :param feature_no:
    :param rp:
    :return:
    """
    # Step 1 : split the data into test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=0.33, random_state=42)

    pipe = Pipeline(params[cp]["Pipeline"])
    hyper_params = params[cp]["hyper_params"]

    # Perform Grid Search
    gridcv = GridSearchCV(pipe,
                          param_grid=hyper_params,
                          cv=5,
                          scoring="accuracy")
    # results
    results = gridcv.fit(X_train, y_train)
    print("Round {} complete for feature set {}".format(cp, feature_no))
    return get_score_data(results, X_train, X_test, y_train, y_test)


def get_feature_sets(X, y, final_feature_set, params):
    """

    :param X:
    :param y:
    :param final_feature_set:
    :return:
    """
    feature_sets = {}
    ll = [x for x in final_feature_set if x in X.columns]
    for cp in params:
        if params[cp]["Pipeline"][0][0] == "classifier":
            estimator = params[cp]["Pipeline"][0][1]
        else:
            estimator = params[cp]["Pipeline"][1][1]
        try:
            feature_selection = SelectFromModel(estimator)
            feature_selection.fit(X, y)
            feature_sets.update({cp: [ll, list(X.columns[feature_selection.get_support()])]})
            print("Feature Selection completed for {}".format(cp))
        except:
            feature_sets.update({cp: [ll, None]})
    return feature_sets


def get_best_scores_params(data, y_col, final_feature_set):
    """

    :param data:
    :param y_col:
    :param final_feature_set:
    :return:
    """
    X = data.drop(columns=[y_col])
    y = data[y_col]
    feature_sets = get_feature_sets(X, y, final_feature_set, classification_params)
    best_scores = {x: {y: [] for y in classification_params} for x in range(1, 3)}

    print("Feature Selection completed..")

    for cp in classification_params:
        for feature_no in range(0, 2):
            features = feature_sets[cp][feature_no]
            if type(features) is list:
                best_scores[feature_no + 1][cp] = model_fit_score(data[features], data[y_col], feature_no + 1, cp, classification_params)

    return best_scores


if __name__ == "__main__":
    pass
