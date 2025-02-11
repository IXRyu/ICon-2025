from dataset_handling import pre_processing
from supervised_training import train_models, test_models
from bayesian import bayesian_classifier
from NNClassifier import NeuralClassifier

def main():
    trained_models = {}
    X_train, X_test, y_train, y_test = pre_processing()

    '''trained_models, best_model = train_models(X_train, y_train, trained_models)
    for model_name, model_info in trained_models.items():
        best_model = model_info['best_estimator']
        test_models(model_name, best_model, X_test, y_test)
    bayesian_classifier()'''
    NeuralClassifier(X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
    main()