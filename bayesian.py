from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from dataset_handling import parting
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

def define_bayesian_network():
    model = BayesianNetwork([
        ('mean_radius', 'mean_area'),
        ('mean_smoothness', 'diagnosis'),
        ('mean_texture', 'diagnosis'),
        ('mean_radius', 'mean_perimeter'),
        ('mean_area', 'mean_perimeter'),
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

def plot(model):
    G = nx.DiGraph()
    for edge in model.edges():
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=7000, node_color='purple', font_size=12, font_weight='bold', arrows=True)
    plt.title("Learned Bayesian Network")
    plt.show()

def bayesian_classifier():
    dataset = parting('dataset/Breast_cancer_data.csv')
    dataset = dataset.dropna()
    dataset = dataset.round(0)
    model = define_bayesian_network()
    #model = learn_structure(dataset)
    model = learn_parameters(model, dataset)
    plot(model)

    samples = random_samples(model, num_samples=1000)
    print("Random samples:\n", samples)

    to_predict = pd.DataFrame({
        'mean_radius': [13,20,15,10,18],
        'mean_perimeter': [120,85,97,70,100],
        'mean_texture': [20,25,27,15,30],
    })

    agent = model.predict(to_predict)
    print("Predicted probabilities:\n", agent)