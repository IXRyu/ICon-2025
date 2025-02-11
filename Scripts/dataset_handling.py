import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from plotter import plot_dataset
import numpy as np
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def augment_dataset(df):
    columns_to_scale = ['mean_radius', 'mean_smoothness', 'mean_perimeter', 'mean_area', 'mean_texture']
    
    scaling_factors = np.linspace(1.1, 2.0, num=5) 
    df_combined = df.copy()
    for factor in scaling_factors:
        df_augmented = df.copy()
        for column in columns_to_scale:
            df_augmented[column] = df_augmented[column] * factor 
        df_combined = pd.concat([df_combined, df_augmented], axis=0, ignore_index=True)
    df_combined.to_csv('./dataset/augmented_data.csv', index=False)
    return df_combined
    
    

def parting(file_path):
    data = pd.read_csv(file_path)
    return data[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'diagnosis']]

def handle_missing_data(df):
    df.dropna(inplace=True)
    return df

def normalization(df):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def pre_processing():
    file_path = './dataset/Breast_cancer_data.csv'
    df = load_data(file_path)
    df = augment_dataset(df)
    df = handle_missing_data(df)
    X, y = normalization(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    plot_dataset(df)
    return X_train, X_test, y_train, y_test