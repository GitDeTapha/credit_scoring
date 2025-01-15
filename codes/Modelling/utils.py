"""
utils_ns.py
   contient les routines communnes et définitions de variable
   Permet de ne pas surcharger le notebook Prediction_No_Show.ipynb
"""


# Chargement des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import PyPDF2
######### 

    #=====================
    #  Tree Ensemble
    #=====================
    
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def evaluate_model(X_train, y_train, X_test, y_test, y_test_proba, y_train_proba, y_test_pred, y_train_pred):
    """
    Evaluate the model by calculating performance metrics with a custom threshold for classification.
    
    Args:
        model: Trained Keras model.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        threshold (float): Threshold to convert probabilities to binary outputs.
    """
   
    # Confusion matrices for the train and test sets
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Display confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion Matrix (Train)")
    ax[0].set_xlabel("Predicted Labels")
    ax[0].set_ylabel("True Labels")
    
    sns.heatmap(cm_test, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion Matrix (Test)")
    ax[1].set_xlabel("Predicted Labels")
    ax[1].set_ylabel("True Labels")
    plt.show()
    # Calculate metrics
    metrics = {
        'Metric': ['Recall', 'AUC', 'Accuracy', 'Precision', 'F1 Score'],
        'Train': [
            recall_score(y_train, y_train_pred),
            roc_auc_score(y_train, y_train_proba),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred)
        ],
        'Test': [
            recall_score(y_test, y_test_pred),
            roc_auc_score(y_test, y_test_proba),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred)
        ]
    }

    # Create a DataFrame for displaying
    results_df = pd.DataFrame(metrics)
    print(results_df.set_index('Metric'))
    return results_df


def evaluate_tree_ensemble_model(model, X_train, y_train, X_test, y_test, y_train_proba, y_test_proba):
    """
    Evaluate the tree ensemble model by calculating performance metrics and displaying both confusion matrices and a table of metrics.

    Args:
        model: Trained tree ensemble model (e.g., RandomForest, XGBoost).
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
    """
    # Predict classes and probabilities
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Display confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion Matrix (Train)")
    ax[0].set_xlabel("Predicted Labels")
    ax[0].set_ylabel("True Labels")
    sns.heatmap(cm_test, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion Matrix (Test)")
    ax[1].set_xlabel("Predicted Labels")
    ax[1].set_ylabel("True Labels")
    plt.show()

    # Calculate metrics
    metrics = {
        'Metric': ['Recall', 'AUC', 'Accuracy', 'Precision', 'F1 Score'],
        'Train': [
            recall_score(y_train, y_train_pred),
            roc_auc_score(y_train, y_train_proba),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred)
        ],
        'Test': [
            recall_score(y_test, y_test_pred),
            roc_auc_score(y_test, y_test_proba),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred)
        ]
    }

    # Create a DataFrame for displaying
    results_df = pd.DataFrame(metrics)
    print(results_df.set_index('Metric'))



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Tracer la courbe ROC pour les prédictions de probabilité d'un modèle.

    Parameters:
    - y_true : array, vrai labels des données de test.
    - y_pred_proba : array, probabilités prédites de la classe positive par le modèle.
    - model_name : str, le nom du modèle pour afficher dans le titre du graphique.

    Returns:
    - Aucun, affiche un graphique.
    """
    # Calcul des taux pour la courbe ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Tracé de la courbe ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


   
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_test, y_test_proba, model_name='Model'):
    """
    Trace la courbe Precision-Recall pour un modèle donné.

    :param y_test: Les valeurs réelles des étiquettes de test
    :param y_test_proba: Les probabilités prédites par le modèle
    :param model_name: Le nom du modèle (par défaut 'Model')
    """
    # Calcul des taux pour la courbe PR
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    average_precision = average_precision_score(y_test, y_test_proba)

    # Tracé de la courbe PR
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AP = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def plot_feature_importances(model, feature_names, title, top_n=16):
    """
    Plot the top n feature importances of the given model.

    Args:
        model: The trained model, e.g., an instance of RandomForestClassifier or RandomForestRegressor.
        feature_names (list): A list of names corresponding to the features in the training data.
        top_n (int): The number of top features to display.
    """
    # Extract feature importances from the model
    importances = model.feature_importances_

    # Create a DataFrame for easier handling
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

    # Sort features according to importance
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    # Select top n features
    feature_importances = feature_importances.head(top_n)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)

    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

from sklearn.utils import resample
# Définition de la fonction plot_feature_importances_bootstrap
def plot_feature_importances_bootstrap(model, X_train, y_train, feature_names, title, top_n=15, n_iterations=10):
    """
    Plot the top n feature importances of the given model using bootstrapping.

    Args:
        model: The trained model, e.g., an instance of RandomForestRegressor or XGBRegressor.
        X_train (DataFrame or ndarray): The training data.
        y_train (Series or ndarray): The target values.
        feature_names (list): A list of names corresponding to the features in the training data.
        top_n (int): The number of top features to display.
        n_iterations (int): The number of bootstrap iterations.
    """
    # Initialize an array to store importances from each iteration
    all_importances = np.zeros((n_iterations + 1, X_train.shape[1]))

    # Store the feature importances of the pre-trained model
    all_importances[0] = model.feature_importances_

    for i in range(1, n_iterations + 1):
        # Create a bootstrap sample
        X_resampled, y_resampled = resample(X_train, y_train)
        
        # Fit the model on the resampled data
        model.fit(X_resampled, y_resampled)
        
        # Store the feature importances
        all_importances[i] = model.feature_importances_
    
    # Compute the mean and standard deviation of feature importances
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    # Create a DataFrame for easier handling
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importances,
        'importance_std': std_importances
    })

    # Sort features according to the mean importance
    feature_importances = feature_importances.sort_values(by='importance_mean', ascending=False)

    # Select top n features
    feature_importances = feature_importances.head(top_n)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance_mean', y='feature', data=feature_importances, xerr=feature_importances['importance_std'])
    plt.title(title)
    plt.xlabel('Importance (mean)')
    plt.ylabel('Features')
    plt.show()# Définition de la fonction plot_feature_importances_bootstrap

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def permutation_importance(model, X, y, metric=accuracy_score):
    baseline = model.evaluate(X, y, verbose=1)
    baseline_metric = baseline[1] 
    importance_scores = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, i] = shuffle(X_permuted[:, i])
        metric_score = model.evaluate(X_permuted, y, verbose=0)[1]
        importance_scores.append(baseline_metric - metric_score)
    
    return importance_scoresb

def plot_boxplot_of_probabilities(model, X_test, y_test, y_test_prob, title, threshold=0.5):
    """
    Trace une boîte à moustaches des probabilités prédites pour les faux positifs et les faux négatifs.

    Parameters:
    - model : le modèle de machine learning
    - X_test : les caractéristiques de test
    - y_test : les étiquettes de test
    - y_test_prob : les probabilités prédites pour les étiquettes de test
    - threshold : le seuil pour déterminer les prédictions positives (par défaut à 0.5)
    """
    # Conversion de y_test en série si ce n'est pas déjà le cas
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()
    elif isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    
    # Vérifier que y_test_prob est de la même longueur que y_test
    if len(y_test) != len(y_test_prob):
        raise ValueError("La longueur de y_test_prob doit être égale à celle de y_test")

    # Détermination des faux positifs et faux négatifs
    fp_probs = y_test_prob[(y_test_prob >= threshold) & (y_test == 0)]
    fn_probs = y_test_prob[(y_test_prob <= threshold) & (y_test == 1)]

    # Création de la boîte à moustaches
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[fp_probs, fn_probs], notch=True, palette=['red', 'blue'])
    plt.xticks([0, 1], ['Faux Positifs', 'Faux Négatifs'])
    plt.ylabel('Probabilités Prédites')
    plt.title(title) #'Distribution des probabilités pour les Faux Positifs et Faux Négatifs'
    plt.show()

def plot_confidence_histogram(model, X_test, y_test, y_test_prob, title, threshold=0.5):
    """
    Trace un histogramme de confiance des probabilités prédites pour les faux positifs et les faux négatifs.

    Parameters:
    - model : le modèle de machine learning
    - X_test : les caractéristiques de test
    - y_test : les étiquettes de test
    - y_test_prob : les probabilités prédites pour les étiquettes de test
    - threshold : le seuil pour déterminer les prédictions positives (par défaut à 0.5)
    """
    # Conversion de y_test en série si ce n'est pas déjà le cas
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()
    elif isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    
    # Vérifier que y_test_prob est de la même longueur que y_test
    if len(y_test) != len(y_test_prob):
        raise ValueError("La longueur de y_test_prob doit être égale à celle de y_test")

    # Détermination des faux positifs et faux négatifs
    fp_probs = y_test_prob[(y_test_prob >= threshold) & (y_test == 0)]
    fn_probs = y_test_prob[(y_test_prob <= threshold) & (y_test == 1)]

    # Création de l'histogramme
    plt.figure(figsize=(12, 8))
    plt.hist(fp_probs, bins=30, alpha=0.5, label='Faux Positifs', color='red')
    plt.hist(fn_probs, bins=30, alpha=0.5, label='Faux Négatifs', color='blue')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Nombre d\'observations')
    plt.title(title)#Histogramme de Confiance des Prédictions
    plt.xlim(0, 1) 
    plt.legend()
    plt.show()
    #===============
    # ANN
    #===============
    
    
    
def permutation_importance(model, X, y, metric=accuracy_score):
    baseline = model.evaluate(X, y, verbose=0)
    baseline_metric = baseline[1]  # Assume that metric of interest is at index 1
    importance_scores = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, i] = shuffle(X_permuted[:, i])
        metric_score = model.evaluate(X_permuted, y, verbose=0)[1]
        importance_scores.append(baseline_metric - metric_score)
    
    return importance_scores
  #===============
    # Common for both ann and tree ensemble
  #===============   
def plot_confidence_histogram(model, X_test, y_test, y_prob):
    # Détermination des faux positifs et faux négatifs
    y_pred = (y_prob > 0.5).astype(int)
    fp_probs = y_prob[(y_pred == 1) & (y_test == 0)]
    fn_probs = y_prob[(y_pred == 0) & (y_test == 1)]

    # Création de l'histogramme
    plt.figure(figsize=(10, 6))
    plt.hist(fp_probs, bins=30, alpha=0.5, label='Faux Positifs', color='red')
    plt.hist(fn_probs, bins=30, alpha=0.5, label='Faux Négatifs', color='blue')
    plt.xlabel('Probabilité Prédite')
    plt.ylabel('Nombre d\'observations')
    plt.title('ANN')
    plt.legend()
    plt.show()
    

def plot_boxplot_of_probabilities(model, X_test, y_test, y_prob):
    # Détermination des faux positifs et faux négatifs
    y_pred = (y_prob > 0.5).astype(int)
    fp_probs = y_prob[(y_pred == 1) & (y_test == 0)]
    fn_probs = y_prob[(y_pred == 0) & (y_test == 1)]

    # Création de la boîte à moustaches
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[fp_probs, fn_probs], notch=True)
    plt.xticks([0, 1], ['Faux Positifs', 'Faux Négatifs'])
    plt.ylabel('Probabilité Prédite')
    plt.title('ANN')
    plt.show()
    
    
    
from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_auc_score, accuracy_score, precision_score, f1_score
def evaluate_tree_ensemble_model(model, X_train, y_train, X_test, y_test, y_train_proba, y_test_proba, threshold=0.5):
    """
    Evaluate the tree ensemble model by calculating performance metrics and displaying both confusion matrices and a table of metrics.

    Args:
        model: Trained tree ensemble model (e.g., RandomForest, XGBoost).
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        threshold: Probability threshold to determine class labels.
    """
    # Convert probabilities to binary predictions based on the threshold
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Display confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion Matrix (Train)")
    ax[0].set_xlabel("Predicted Labels")
    ax[0].set_ylabel("True Labels")
    sns.heatmap(cm_test, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion Matrix (Test)")
    ax[1].set_xlabel("Predicted Labels")
    ax[1].set_ylabel("True Labels")
    plt.show()

    # Calculate metrics
    metrics = {
        'Metric': ['Recall', 'AUC', 'Accuracy', 'Precision', 'F1 Score'],
        'Train': [
            recall_score(y_train, y_train_pred),
            roc_auc_score(y_train, y_train_proba),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred)
        ],
        'Test': [
            recall_score(y_test, y_test_pred),
            roc_auc_score(y_test, y_test_proba),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred)
        ]
    }

    # Create and print a DataFrame for displaying metrics
    results_df = pd.DataFrame(metrics)
    print(results_df.set_index('Metric'))
    return results_df


def plot_learning_curve_ann(history):
    # Tracé de la précision
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Tracé de la perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.show()    
