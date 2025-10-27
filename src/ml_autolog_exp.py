import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import pickle

dagshub.init(repo_owner='sharadkrishna510', repo_name='MLFlow_experiments', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sharadkrishna510/MLFlow_experiments.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 15
n_estimators = 4

#auto logging enabled
mlflow.autolog()

# Mention your experiment below
mlflow.set_experiment('MLOPS-mlflow-Exp1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric('accuracy', accuracy)
    # mlflow.log_param('max_depth', max_depth)
    # mlflow.log_param('n_estimators', n_estimators)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    # mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Sharad', "Project": "Wine Classification"})

    # # Log the model
    # with open("Random-Forest-Model.pkl", "wb") as f:
    #     pickle.dump(rf, f)

    # # Log the pickle file as artifact (DagsHub supports this)
    # mlflow.log_artifact("Random-Forest-Model.pkl")

    print(accuracy)