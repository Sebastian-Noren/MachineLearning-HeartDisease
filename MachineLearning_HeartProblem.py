# * Import all Tools I need *

# Regular EDA (Exploratory data analysis) and plotting libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Models building from Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# * LOAD and EVALUATE Data *

heartData = pd.read_csv("heart_problem_data.csv")

# Data Exploration (Data analysis/EDA)
print(heartData.head())
# target - Heart problem or not (1= yes, 0 = no)
print(heartData["target"].value_counts())
heartData["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.show()
# Check data entry
print(heartData.info())
# Check if values is missing ( No missing)
print(heartData.isna().sum())

# Compare target (heart problem) with sex (women = 0, men = 1)
print(pd.crosstab(heartData.target, heartData.sex))
pd.crosstab(heartData.target, heartData.sex).plot(kind="bar", figsize=(10, 6),
                                                  color=["salmon", "lightblue"])
plt.title("Heart Disease Freq. for Sex")
plt.xlabel("0 = No Heart problem, 1 = Heart problem")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0)
plt.show()

# Age vs Max Heart rate dor Heart problem ( scatter diagram)
plt.figure(figsize=(10, 6))
# scatter WITH heart problems (positive examples)
plt.scatter(heartData.age[heartData.target == 1],
            heartData.thalach[heartData.target == 1], c="red")

# scatter WITHOUT heart problems (positive examples)
plt.scatter(heartData.age[heartData.target == 0],
            heartData.thalach[heartData.target == 0], c="blue")

plt.title("Heart Disease in function of Age and Max heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Heart Problem", "Healthy"])
# Save figure
figure = plt
figure.savefig("heart-scatter.png")
# display figure
plt.show()

# * CREATE MACHINE LEARNING MODEL *

# Split the data into X and Y (binary classification)
X = heartData.drop("target", axis=1)  # Drop target column
Y = heartData["target"]  # assign target value to Y
# Split data into train and test sets. 80% train data, 20% test data
np.random.seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Test 3 machine learning models: Logistic Regression, K-nearest Neighbours classifier, Random Forrest classifier

# Put Models in a dictionary
models = {"LogisticRegression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "RandomForest": RandomForestClassifier()}


# Function to fit and score models
def fit_and_score(modelsToTest, x_train, x_test, y_train, y_test):
    np.random.seed(42)
    # Make a dictionary keep model scores
    models_scores = {}
    for name, model in modelsToTest.items():
        # Fit the model to the data
        model.fit(x_train, y_train)
        # Evaluate the model and append its score
        models_scores[name] = model.score(x_test, y_test)
    return models_scores


testScores = fit_and_score(models, X_train, X_test, Y_train, Y_test)

print(testScores)
model_compare = pd.DataFrame(testScores, index=["Accuracy"])
model_compare.T.plot.bar()
plt.show()

# * MODEL TUNING and IMPROVEMENT *

# Hyperparameter tuning KNN By Hand
train_scores = []
test_scores = []
# Create a list different values for n_neighbors
neighbors = range(1, 21)
# Setup KNN instance
knn = KNeighborsClassifier()
# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    # fit the algorithm
    knn.fit(X_train, Y_train)
    # update the training scores list
    train_scores.append(knn.score(X_train, Y_train))
    test_scores.append(knn.score(X_test, Y_test))

plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test Score")
plt.xlabel("Number of neighbors")
plt.xticks(np.arange(1, 21, 1))
plt.ylabel("Model score")
plt.legend()
plt.show()
print(f"Maximum KNN Score on the test data: {max(test_scores) * 100:.2f}%")
# KNN Still Bad score after tuning, DISCARD KNN!

# - HyperParameter with RandomizedSearchCV -

# Create a HyperParameter grid for LogicRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a HyperParameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

# Hyperparameter grids setup for both models, tune models using RandomizedSearchCV
np.random.seed(42)
# Cross validation number (K-fold cross-validation)
kFoldCV = 5
# Setup random hypePar for LogReg
log_reg_rs = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid,
                                cv=kFoldCV, n_iter=20, verbose=True)
# Fit random hyperParameter to the model
log_reg_rs.fit(X_train, Y_train)
# Best params for Logic Regression
print(log_reg_rs.best_params_)
print(log_reg_rs.score(X_test, Y_test))

# Setup random hypePar for Random Forrest
np.random.seed(42)
rf_rs = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=kFoldCV, n_iter=20, verbose=True)
# # Fit random hyperParameter to Random Forest
rf_rs.fit(X_train, Y_train)
# # Best params for Random Forest
print(rf_rs.best_params_)
print(rf_rs.score(X_test, Y_test))
# Random Forest will be Discarded

# - Final Tuning last model with GridSearchCV -

# Tune hyperPara for LogReg
log_reg_grid = {"C": np.logspace(-4, 4, 40),
                "solver": ["liblinear"]}

# Setup grid hypePar search for LogReg
log_reg_gs = GridSearchCV(LogisticRegression(), param_grid=log_reg_grid,
                          cv=kFoldCV, verbose=True)
# Fit Grid hyperParameter to the model
log_reg_gs.fit(X_train, Y_train)

tunedModel = log_reg_gs

# * EVALUATING THE TUNED MACHINE LEARNING MODEL *

# Make prediction with tuned model
y_predictions = tunedModel.predict(X_test)

# Plot ROC curve and Calculate AUC metric
plot_roc_curve(tunedModel, X_test, Y_test)
plt.show()


# *********** FUNCTION ******************

# Set up Confusion matrix function
def plot_conf_mat(y_test, y_preds):
    sb.set(font_scale=1.5)
    plt.figure(figsize=(3, 3))
    plt.subplot()
    sb.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()


# ****************************************


# Display it
# true positive = model predicts 1 when truth is 1
# False positive = model predicts 1 when truth is 0
# True negative = model predicts 0 when truth is 0
# False Negative = model predicts 0 when truth is 1 (not heart problem, when person have a heart problem)
plot_conf_mat(Y_test, y_predictions)

# - Calculate evaluation metrics with cross-validation -
# accuracy, Precision, recall and f1-score for the model

# Check best hyperPara for tuned model
print(tunedModel.best_params_)
# Create new classifier with the models best parameters
CLF = LogisticRegression(C=0.19144819761699575, solver="liblinear")


# *********** FUNCTION ******************

def cross_validation(clf, x, y, k_fold, test_string):
    cv = cross_val_score(clf, x, y, cv=k_fold, scoring=test_string)
    cv = np.mean(cv)
    return cv


# ****************************************

# cross-validated accuracy
cv_acc = cross_validation(CLF, X, Y, kFoldCV, "accuracy")
print(f"Accuracy: {cv_acc * 100:.2f}%")
# cross-validated precision
cv_precision = cross_validation(CLF, X, Y, kFoldCV, "precision")
print(f"Precision: {cv_precision * 100:.2f}%")
# cross-validated recall
cv_recall = cross_validation(CLF, X, Y, kFoldCV, "recall")
print(f"Recall: {cv_recall * 100:.2f}%")
# cross-validated f1-score
cv_f1 = cross_validation(CLF, X, Y, kFoldCV, "f1")
print(f"F1-score: {cv_f1 * 100:.2f}%")

# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1}, index=[0])
cv_metrics.T.plot.bar(title="Cross Validated classification")
plt.show()

# - Finding the most important features (model specific)
CLF.fit(X_train, Y_train)
# Match coef's for a feature to it's columns
feature_dict = dict(zip(heartData.columns, list(CLF.coef_[0])))
# visualize feature importance
feature_visualization = pd.DataFrame(feature_dict, index=[0])
feature_visualization.T.plot.bar(title="Feature Importance", legend=False)
plt.show()
