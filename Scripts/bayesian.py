from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from plotter import plot_bayesian

from dataset_handling import parting

def define_bayesian_network():
    model = BayesianNetwork([
        ('mean_radius', 'mean_perimeter'),
        ('mean_smoothness', 'diagnosis'),
        ('mean_texture', 'diagnosis'),

        ('mean_perimeter', 'mean_area'),
        ('mean_radius', 'diagnosis'),
        ('mean_area', 'diagnosis'),
        ('mean_perimeter', 'diagnosis')
    ])
    return model
    
def learn_structure(data):
    estimator = HillClimbSearch(data)
    model = estimator.estimate(scoring_method=K2Score(data), max_indegree=5, max_iter=int(1e5))
    return model

def learn_parameters(model, data):
    bayesian_model = BayesianNetwork(model.edges())
    bayesian_model.fit(data, estimator=MaximumLikelihoodEstimator)
    return bayesian_model

def random_samples(model, num_samples):
    samples = model.simulate(num_samples)
    diagnosis_counts = samples['diagnosis'].value_counts()
    print("Diagnosis counts:\n", diagnosis_counts)
    return samples

def bayesian_classifier():
    dataset = parting('./dataset/Breast_cancer_data.csv')
    dataset = dataset.dropna()
    #discretizzo il dataset

    '''discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    continuous_columns = ['mean_smoothness']
    dataset[continuous_columns] = discretizer.fit_transform(dataset[continuous_columns])'''


    dataset = dataset.round(0) # Ho provato a discretizzare i dati ma comunque non ne ho la RAM sufficiente
    #model = define_bayesian_network()
    model = learn_structure(dataset)
    model = learn_parameters(model, dataset)
    plot_bayesian(model)

    samples = random_samples(model, num_samples=100)
    print("Random samples:\n", samples)

    to_predict = pd.DataFrame({
        'mean_radius': [13, 20,15],
        'mean_perimeter': [120, 85, 97],
        'mean_texture': [13, 23, 15],
    })

    agent = model.predict(to_predict)
    print("Predicted probabilities:\n", agent)