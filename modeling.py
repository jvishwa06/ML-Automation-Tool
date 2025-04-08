import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBClassifier, XGBRegressor
import shap
import time
import warnings

warnings.filterwarnings('ignore')

def train_models(X_train, y_train, task_type):
    """
    Train various machine learning models based on the task type
    """
    models = {}
    
    if task_type == 'classification':
        # classification
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVC': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(random_state=42, verbosity=0)
        }
    else:  # regression
        models = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'XGBoost': XGBRegressor(random_state=42, verbosity=0)
        }
    
    trained_models = {}
    for name, model in models.items():
        if X_train.shape[0] > 10000 and name in ['SVC', 'SVR']:
            continue
            
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"Trained {name} in {training_time:.2f} seconds")
            trained_models[name] = model
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
    
    return trained_models

def evaluate_models(trained_models, X_train, X_test, y_train, y_test, task_type, feature_names):
    """
    Evaluate trained models and generate performance metrics and visualizations
    """
    evaluation_results = {}
    model_plots = {}
    
    performance_metrics = {}
    
    if task_type == 'classification':
        for name, model in trained_models.items():
            try:
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)
                precision = precision_score(y_test, y_pred_test, average='weighted')
                recall = recall_score(y_test, y_pred_test, average='weighted')
                f1 = f1_score(y_test, y_pred_test, average='weighted')
                
                cm = confusion_matrix(y_test, y_pred_test)
                
                evaluation_results[name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'performance_metric': f1  
                }
                performance_metrics[name] = test_acc
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
    
    else:  # regression
        for name, model in trained_models.items():
            try:
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                
                try:
                    mape = mean_absolute_percentage_error(y_test, y_pred_test)
                except:
                    mape = np.mean(np.abs((y_test - y_pred_test) / np.maximum(np.ones(len(y_test)), np.abs(y_test)))) * 100
                
                evaluation_results[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'mape': mape,
                    'performance_metric': r2  
                }
                performance_metrics[name] = r2
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
    
    # Generate model comparison plot
    plt.figure(figsize=(12, 6))
    if task_type == 'classification':
        metrics_to_plot = ['test_accuracy', 'precision', 'recall', 'f1_score']
        title = 'Classification Model Performance Comparison'
        ylabel = 'Score'
    else:
        metrics_to_plot = ['test_rmse', 'mae', 'r2_score']
        title = 'Regression Model Performance Comparison'
        ylabel = 'Score/Error'
    
    plot_data = []
    for model_name, metrics in evaluation_results.items():
        for metric in metrics_to_plot:
            if metric in metrics:
                plot_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': metrics[metric]
                })
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Model', y='Value', hue='Metric', data=plot_df)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        model_plots['model_comparison'] = plt.gcf()
        plt.close()
    
    # Generate feature importance plot (using the best model)
    if performance_metrics:
        best_model_name = max(performance_metrics, key=performance_metrics.get)
        best_model = trained_models[best_model_name]
        
        try:
            plt.figure(figsize=(12, 8))
            
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                if len(feature_names) < len(importances):
                    display_names = [f"Feature {i}" for i in range(len(importances))]
                else:
                    display_names = feature_names
                
                n_features = min(20, len(importances))
                plt.figure(figsize=(12, 8))  
                plt.bar(range(n_features), importances[indices[:n_features]])
                plt.xticks(range(n_features), [display_names[i] if i < len(display_names) else f"Feature {i}" for i in indices[:n_features]], rotation=90)
                plt.title(f'Feature Importance - {best_model_name}')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                model_plots['feature_importance'] = plt.gcf()
                plt.close()
            
            # SHAP feature importance for more complex models
            elif best_model_name in ['XGBoost', 'RandomForest', 'GradientBoosting']:
                sample_size = min(100, X_test.shape[0])
                X_sample = X_test[:sample_size]
                
                try:
                    explainer = shap.Explainer(best_model)
                    shap_values = explainer(X_sample)
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, feature_names=feature_names, plot_type='bar', show=False)
                    plt.title(f'SHAP Feature Importance - {best_model_name}')
                    plt.tight_layout()
                    model_plots['feature_importance'] = plt.gcf()
                    plt.close()
                except Exception as e:
                    print(f"Error creating SHAP plot: {str(e)}")
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")
    
    return evaluation_results, model_plots
