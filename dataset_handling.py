import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

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
    file_path = 'dataset/Breast_cancer_data.csv'
    df = load_data(file_path)
    df = handle_missing_data(df)
    X, y = normalization(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test