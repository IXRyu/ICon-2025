import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from plotter import plot_NN_curves

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x) 
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)  
        x = torch.sigmoid(self.output(x))  # Uscita con sigmoid per la classificazione binaria
        return x

def Training_Prediction(X_train, y_train, X_test, y_test, epochs=100, patience=5, batch_size=64, lr=0.001, dropout_rate=0.5):
    # Normalizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Conversione in tensori torch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Creazione del dataset e DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Inizializzazione del modello, della loss e dell'ottimizzatore
    model = NeuralNetwork(input_dim=X_train.shape[1], dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)  # L2 Regularization

    def compute_metrics(X, y):
        model.eval()
        with torch.no_grad():
            y_pred = model(X)
            y_pred = (y_pred > 0.5).float()
            accuracy = accuracy_score(y.numpy(), y_pred.numpy())
            precision = precision_score(y.numpy(), y_pred.numpy())
            recall = recall_score(y.numpy(), y_pred.numpy())
            f1 = f1_score(y.numpy(), y_pred.numpy())
        return accuracy, precision, recall, f1

    # Funzione di early stopping
    def early_stopping(patience, val_loss_history):
        if len(val_loss_history) < patience:
            return False
        return val_loss_history[-1] > min(val_loss_history[-patience:])

    history = {
        'accuracy': [], 'val_accuracy': [],
        'precision': [], 'val_precision': [],
        'recall': [], 'val_recall': [],
        'f1': [], 'val_f1': []
    }

    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(X_train_tensor, y_train_tensor)
        val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(X_test_tensor, y_test_tensor)

        history['accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Memorizzare il val_loss per Early Stopping
        val_loss = running_loss / len(train_loader)
        val_loss_history.append(val_loss)

        # Controlla se applicare Early Stopping
        '''if early_stopping(patience, val_loss_history):
            print("Early stopping triggered!")
            break'''

    test_accuracy, test_precision, test_recall, test_f1 = compute_metrics(X_test_tensor, y_test_tensor)
    return {
        'history': history,
        'test_metrics': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        },
        'model': model
    }

def NeuralClassifier(X_train, y_train, X_test , y_test):
    results = Training_Prediction(X_train, y_train, X_test, y_test, epochs=100, patience=5)
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.8f}")
    print(f"Test Precision: {results['test_metrics']['precision']:.8f}")
    print(f"Test Recall: {results['test_metrics']['recall']:.8f}")
    print(f"Test F1-Score: {results['test_metrics']['f1']:.8f}")
    plot_NN_curves(results['history'])