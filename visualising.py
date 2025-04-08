import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_plots(df, target_column, task_type):
    """
    Generate exploratory data analysis plots
    """
    plots = []
    
    # Plot 1: Missing values heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plots.append(plt.gcf())
    plt.close()
    
    # Plot 2: Distribution of target variable
    plt.figure(figsize=(10, 6))
    if task_type == 'classification':
        target_counts = df[target_column].value_counts()
        plt.bar(target_counts.index.astype(str), target_counts.values)
        plt.title(f'Distribution of {target_column}')
        plt.xticks(rotation=45)
    else: 
        plt.hist(df[target_column].dropna(), bins=30, alpha=0.7)
        plt.title(f'Distribution of {target_column}')
    plt.tight_layout()
    plots.append(plt.gcf())
    plt.close()
    
    # Plot 3: Correlation heatmap for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:  
        plt.figure(figsize=(12, 10))
        correlation = numeric_df.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plots.append(plt.gcf())
        plt.close()
    
    # Plot 4: Box plots for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_melt = pd.melt(df[numeric_cols], var_name='Feature', value_name='Value')
        sns.boxplot(x='Feature', y='Value', data=df_melt)
        plt.title('Box Plots for Numeric Features')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plots.append(plt.gcf())
        plt.close()
    
    # Plot 5: Target vs Top Features (for classification)
    if task_type == 'classification':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if numeric_cols:
            correlations = []
            for col in numeric_cols:
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    corr = df[col].corr(df[target_column])
                else:
                    from scipy import stats
                    groups = df.groupby(target_column)[col].apply(list)
                    fvalue, pvalue = stats.f_oneway(*groups)
                    corr = fvalue
                correlations.append((col, abs(corr)))
            
            top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:5]
            top_feature_names = [f[0] for f in top_features]
            
            for feature in top_feature_names:
                plt.figure(figsize=(10, 6))
                for target_val in df[target_column].unique():
                    sns.histplot(df[df[target_column] == target_val][feature], 
                                kde=True, alpha=0.5, label=str(target_val))
                plt.title(f'{feature} Distribution by {target_column}')
                plt.legend()
                plt.tight_layout()
                plots.append(plt.gcf())
                plt.close()
    
    # Plot 6: Scatter plot matrix for top correlated features (regression)
    if task_type == 'regression' and df.select_dtypes(include=[np.number]).shape[1] > 1:
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corrwith(df[target_column]).abs().sort_values(ascending=False)
        top_features = correlations.index[:min(5, len(correlations))].tolist()
        if target_column in top_features:
            top_features.remove(target_column)
        
        if top_features:
            top_features.append(target_column)
            plt.figure(figsize=(12, 10))
            sns.pairplot(df[top_features], hue=target_column if task_type == 'classification' else None)
            plt.tight_layout()
            plots.append(plt.gcf())
            plt.close()
    
    return plots
