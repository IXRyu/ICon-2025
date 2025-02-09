import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import learning_curve
import numpy as np

def plot_training_report(model, X_train, y_train, model_name):
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
    
    cm = confusion_matrix(y_train, y_train_pred)
    precision = precision_score(y_train, y_train_pred)
    recall = recall_score(y_train, y_train_pred)
    f1 = f1_score(y_train, y_train_pred)
    accuracy = accuracy_score(y_train, y_train_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title(f'Matrice di Confusione - {model_name} (Train)')
    plt.show()
    
    plt.figure(figsize=(6, 4))
    metrics = [accuracy, precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    sns.barplot(x=metric_names, y=metrics, palette='Blues')
    plt.title(f'Metriche di Valutazione - {model_name} (Train)')
    plt.ylim(0, 1)
    plt.show()
    
    if y_train_prob is not None:
        fpr, tpr, _ = roc_curve(y_train, y_train_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {model_name} (Train)')
        plt.legend(loc='lower right')
        plt.show()
        
        precision, recall, _ = precision_recall_curve(y_train, y_train_prob)
        ap = average_precision_score(y_train, y_train_prob)
        
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name} (Train)')
        plt.legend(loc='upper right')
        plt.show()

def plot_test_report(model, model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title(f'Matrice di Confusione - {model_name} (Test)')
    plt.show()
    
    plt.figure(figsize=(6, 4))
    metrics = [accuracy, precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    sns.barplot(x=metric_names, y=metrics, palette='Blues')
    plt.title(f'Metriche di Valutazione - {model_name} (Test)')
    plt.ylim(0, 1)
    plt.show()
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(y_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {model_name} (Test)')
        plt.legend(loc='lower right')
        plt.show()
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name} (Test)')
        plt.legend(loc='upper right')
        plt.show()

def plot_learning_curve(model, X, y, model_name):
    scoring_metrics = ['f1', 'precision', 'recall']
    titles = ['F1 Error', 'Precision Error', 'Recall Error']
    colors = [('purple', 'lightblue'), ('red', 'darkred'), ('green', 'lightgreen')]

    plt.figure(figsize=(18, 8))

    for i, scoring in enumerate(scoring_metrics):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring=scoring, n_jobs=-1)

        train_errors = 1 - train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_errors = 1 - test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)

        train_sizes_percent = (train_sizes / len(X)) * 100

        plt.subplot(2, 3, i + 1)
        plt.plot(train_sizes_percent, train_errors, label="Train error", color=colors[i][0])
        plt.fill_between(train_sizes_percent, train_errors - train_std, train_errors + train_std, color=colors[i][1], alpha=0.3)
        plt.plot(train_sizes_percent, test_errors, label="Test error", color=colors[i][0], linestyle='dashed')
        plt.fill_between(train_sizes_percent, test_errors - test_std, test_errors + test_std, color=colors[i][1], alpha=0.3)

        plt.title(f'{titles[i]} ({model_name})')
        plt.xlabel('Training Set Size')
        plt.ylabel('Error')
        plt.legend(loc="best")
        plt.grid()

        

        plt.subplot(2, 3, i + 4)
        plt.axis('tight')
        plt.axis('off')

    plt.tight_layout()
    plt.show()