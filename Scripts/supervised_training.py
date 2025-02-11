from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from plotter import plot_test_report, plot_training_report, plot_learning_curve

def get_parameters():
    return {
        'Decision_Tree': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5, 10],
        },
        'Random_Forest': {
            'n_estimators': [25, 50],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10],
        },
        'Logistic_Regression': {
            'penalty': ['l1','l2'],
            'C': [0.001, 0.01, 0.1],
            'solver': ['liblinear'],
            'max_iter': [100000, 150000],
        },
        'SVM': {
            'C': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
        },
        'k-NN': {
            'n_neighbors': [5],
            'weights': ['distance'],  
            'algorithm': ['auto'], 
            'p': [2],  
            'n_jobs': [1]  
        },
    }

def init_models():
    return {
        'Decision_Tree': DecisionTreeClassifier(),
        'Logistic_Regression': LogisticRegression(),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(),
        'k-NN': KNeighborsClassifier()
    }

def cross_validation(name, model, x_train, y_train, params):
    x_train_half, _, y_train_half, _ = train_test_split(x_train, y_train, test_size=0.6, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=10, n_jobs=-1, verbose=0)
    grid_search.fit(x_train_half, y_train_half)
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Validation accuracy for {name}: {grid_search.best_score_:.4f}\n")
    return {
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'validation_score': grid_search.best_score_,
    }

def test_models(name, model, X_test, y_test):
    print(f"Testing {name}...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    plot_test_report(report, name, y_test, y_pred)
    print(f"Classification report for {name} on test set:\n{report}")
    return report

def train_models(X, y, trained_models):
    params = get_parameters()
    models = init_models()
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        trained_models[model_name] = cross_validation(model_name, model, X, y, params[model_name])

    for model_name, model_info in trained_models.items():
        best_model = model_info['best_estimator']
        plot_training_report(best_model, X, y , model_name)
        plot_learning_curve(best_model, X, y, model_name)
    return trained_models, best_model