import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from preprocessing import preprocess_data
from visualising import generate_eda_plots
from modeling import train_models, evaluate_models

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def process_csv(file, target_column, task_type, test_size, random_state):
    df = pd.read_csv(file.name)
    
    info = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Data Types": df.dtypes.astype(str).to_dict(),
        "Missing Values": df.isna().sum().to_dict(),
        "Numeric Columns": list(df.select_dtypes(include=[np.number]).columns),
        "Categorical Columns": list(df.select_dtypes(include=['object', 'category']).columns)
    }
    
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        df, target_column, task_type, test_size, random_state
    )
    
    eda_plots = generate_eda_plots(df, target_column, task_type)
    
    trained_models = train_models(X_train, y_train, task_type)
    
    evaluation_results, model_plots = evaluate_models(
        trained_models, X_train, X_test, y_train, y_test, task_type, feature_names
    )
    
    model_paths = {}
    for model_name, model in trained_models.items():
        path = f"models/{model_name}.joblib"
        joblib.dump(model, path)
        model_paths[model_name] = path
    
    results = {
        "info": info,
        "eda_plots": eda_plots,
        "evaluation_results": evaluation_results,
        "model_plots": model_plots,
        "model_paths": model_paths,
        "best_model": max(evaluation_results, key=lambda x: evaluation_results[x]['performance_metric'])
    }
    
    return results

def create_custom_plot(plot_type, selected_columns, hue_column, file):
    if not file or not selected_columns:
        return None
        
    df = pd.read_csv(file.name)
    
    plt.figure(figsize=(10, 6))
    
    try:
        if plot_type == "Histogram":
            if len(selected_columns) > 1:
                fig, axes = plt.subplots(1, len(selected_columns), figsize=(5*len(selected_columns), 5))
                for i, col in enumerate(selected_columns):
                    if hue_column and hue_column in df.columns:
                        for cat in df[hue_column].unique():
                            subset = df[df[hue_column] == cat]
                            axes[i].hist(subset[col], alpha=0.5, label=str(cat))
                        axes[i].set_title(f'Histogram of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].legend()
                    else:
                        axes[i].hist(df[col], bins=20)
                        axes[i].set_title(f'Histogram of {col}')
                        axes[i].set_xlabel(col)
            else:
                col = selected_columns[0]
                if hue_column and hue_column in df.columns:
                    for cat in df[hue_column].unique():
                        subset = df[df[hue_column] == cat]
                        plt.hist(subset[col], alpha=0.5, label=str(cat))
                    plt.title(f'Histogram of {col} by {hue_column}')
                    plt.xlabel(col)
                    plt.legend()
                else:
                    plt.hist(df[col], bins=20)
                    plt.title(f'Histogram of {col}')
                    plt.xlabel(col)
        
        elif plot_type == "Box Plot":
            if hue_column and hue_column in df.columns:
                ax = sns.boxplot(data=df, x=hue_column, y=selected_columns[0] if len(selected_columns) == 1 else None)
                plt.title(f'Box Plot of {selected_columns[0]} by {hue_column}')
            else:
                ax = sns.boxplot(data=df[selected_columns])
                plt.title(f'Box Plot of {", ".join(selected_columns)}')
        
        elif plot_type == "Bar Chart":
            if len(selected_columns) == 1:
                if df[selected_columns[0]].nunique() < 15:  # Only for categorical with reasonable number of values
                    counts = df[selected_columns[0]].value_counts()
                    sns.barplot(x=counts.index, y=counts.values)
                    plt.title(f'Bar Chart of {selected_columns[0]}')
                    plt.xticks(rotation=45)
            else:
                if hue_column and hue_column in df.columns:
                    df_melt = pd.melt(df, id_vars=[hue_column], value_vars=selected_columns)
                    sns.barplot(x='variable', y='value', hue=hue_column, data=df_melt)
                    plt.title(f'Bar Chart by {hue_column}')
                else:
                    df_melt = pd.melt(df, value_vars=selected_columns)
                    sns.barplot(x='variable', y='value', data=df_melt)
                    plt.title(f'Bar Chart of Selected Columns')
        
        elif plot_type == "Line Chart":
            if len(selected_columns) > 0:
                for col in selected_columns:
                    plt.plot(df.index, df[col], label=col)
                plt.title('Line Chart')
                plt.legend()
            
        elif plot_type == "Scatter Plot":
            if len(selected_columns) >= 2:
                if hue_column and hue_column in df.columns:
                    sns.scatterplot(data=df, x=selected_columns[0], y=selected_columns[1], hue=hue_column)
                else:
                    sns.scatterplot(data=df, x=selected_columns[0], y=selected_columns[1])
                plt.title(f'Scatter Plot: {selected_columns[0]} vs {selected_columns[1]}')
            
        elif plot_type == "Correlation Heatmap":
            if len(selected_columns) > 1:
                corr = df[selected_columns].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation Heatmap')
            else:
                plt.text(0.5, 0.5, "Select at least 2 columns for correlation heatmap", 
                        ha='center', va='center', fontsize=12)
        
        elif plot_type == "Pair Plot":
            if len(selected_columns) > 1:
                if hue_column and hue_column in df.columns:
                    pair_fig = sns.pairplot(df[selected_columns + [hue_column]], hue=hue_column)
                    return pair_fig
                else:
                    pair_fig = sns.pairplot(df[selected_columns])
                    return pair_fig
            else:
                plt.text(0.5, 0.5, "Select at least 2 columns for pair plot", 
                        ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        return plt.gcf()
    
    except Exception as e:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                ha='center', va='center', fontsize=12)
        return plt.gcf()
    finally:
        plt.close()

def create_ui():
    with gr.Blocks(title="Machine Learning Automation") as demo:
        gr.Markdown("# Machine Learning Automation Tool")
        gr.Markdown("Upload your CSV data, and this tool will automatically preprocess it, generate visualizations, train various models, and provide evaluation metrics.")
        
        column_names = gr.State([])
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload CSV Data")
                target_column = gr.Dropdown(
                    label="Target Column Name",
                    choices=[],
                    interactive=True
                )
                task_type = gr.Dropdown(
                    label="Task Type", 
                    choices=["classification", "regression"], 
                    value="classification"
                )

                test_size = gr.Slider(
                    label="Test Size", 
                    minimum=0.1, 
                    maximum=0.5, 
                    value=0.2, 
                    step=0.05
                )
                random_state = gr.Number(
                    label="Random State", 
                    value=42, 
                    precision=0
                )
                submit_button = gr.Button("Process Data and Train Models")
            
            with gr.Column():
                with gr.Tab("Data Info"):
                    with gr.Row():
                        with gr.Column():
                            dataset_shape = gr.Textbox(label="Dataset Shape", interactive=False)
                            missing_values = gr.Dataframe(label="Missing Values Summary", interactive=False)
                        with gr.Column():
                            data_sample = gr.Dataframe(label="Data Sample", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            numeric_stats = gr.Dataframe(label="Numeric Columns Statistics", interactive=False)
                        with gr.Column():
                            categorical_stats = gr.Dataframe(label="Categorical Columns Summary", interactive=False)
                
                with gr.Tab("EDA Visualizations"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            plot_type = gr.Radio(
                                label="Plot Type",
                                choices=["Histogram", "Box Plot", "Bar Chart", "Line Chart", "Scatter Plot", "Correlation Heatmap", "Pair Plot"],
                                value="Histogram"
                            )
                            column_selector = gr.Dropdown(
                                label="Select Column(s)",
                                multiselect=True,
                                interactive=True
                            )
                            generate_plot_btn = gr.Button("Generate Plot")
                        with gr.Column(scale=2):
                            eda_plots_output = gr.Plot(label="Exploratory Data Analysis")
                
                with gr.Tab("Model Evaluation"):
                    model_metrics = gr.DataFrame(label="Model Performance Metrics")
                    model_plot = gr.Plot(label="Performance Visualization")
                
                with gr.Tab("Feature Importance"):
                    feature_imp_plot = gr.Plot(label="Feature Importance")
                
                with gr.Tab("Download Models"):
                    model_select = gr.Dropdown(label="Select Model to Download")
                    download_btn = gr.Button("Download Selected Model")
                    download_output = gr.File(label="Download")
        
        def update_columns(file):
            if not file:
                return gr.Dropdown(choices=[])
            
            df = pd.read_csv(file.name)
            columns = df.columns.tolist()
            return gr.Dropdown(choices=columns)
        
        def process_and_display(file, target, task, test_sz, rand_state):
            if not file:
                return (None, None, None, None, None, None, None, 
                        None, None, None, gr.Dropdown(choices=[]))
            
            df = pd.read_csv(file.name)
            
            results = process_csv(file, target, task, test_sz, rand_state)
            
            shape_info = f"{df.shape[0]} rows × {df.shape[1]} columns"
            
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isna().sum().values,
                'Missing Percentage': (df.isna().sum().values / len(df) * 100).round(2)
            })
            
            sample_df = df.head(5)
            
            numeric_df = df.select_dtypes(include=[np.number]).describe().T.reset_index()
            numeric_df = numeric_df.rename(columns={'index': 'Column'})
            
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            cat_summary = []
            for col in cat_cols:
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Top Value': df[col].value_counts().index[0] if not df[col].isna().all() else "N/A",
                    'Frequency': df[col].value_counts().values[0] if not df[col].isna().all() else 0,
                    'Frequency %': (df[col].value_counts().values[0] / len(df) * 100).round(2) if not df[col].isna().all() else 0
                })
            categorical_df = pd.DataFrame(cat_summary) if cat_summary else pd.DataFrame(columns=['Column', 'Unique Values', 'Top Value', 'Frequency', 'Frequency %'])
            
            column_choices = df.columns.tolist()
            
            metrics_df = pd.DataFrame(
                {model: {k: round(v, 4) if isinstance(v, (int, float)) else v 
                        for k, v in metrics.items() if k != 'performance_metric'}
                 for model, metrics in results["evaluation_results"].items()}
            ).T.reset_index().rename(columns={"index": "Model"})
            
            model_comparison = results["model_plots"].get("model_comparison", None)
            
            feature_imp = results["model_plots"].get("feature_importance", None)
            
            model_names = list(results["model_paths"].keys())
            
            return (shape_info, missing_df, sample_df, numeric_df, categorical_df, 
                    gr.Dropdown(choices=column_choices, multiselect=True), 
                    gr.Dropdown(choices=column_choices), 
                    None, metrics_df, model_comparison, feature_imp, 
                    gr.Dropdown(choices=model_names))
        
        def update_eda_plot(chart_index, file, target, task, test_sz, rand_state):
            if not file or chart_index is None:
                return None
                
            results = process_csv(file, target, task, test_sz, rand_state)
            eda_plots = results["eda_plots"]
            
            if not eda_plots or int(chart_index.split()[-1]) > len(eda_plots):
                return None
                
            return eda_plots[int(chart_index.split()[-1]) - 1]
        
        def download_model(model_name):
            if not model_name:
                return None
            
            model_path = f"models/{model_name}.joblib"
            if not os.path.exists(model_path):
                return None
            
            return model_path
        
        file_input.change(
            update_columns,
            inputs=[file_input],
            outputs=[target_column]
        )
        
        submit_button.click(
            process_and_display,
            inputs=[file_input, target_column, task_type, test_size, random_state],
            outputs=[dataset_shape, missing_values, data_sample, numeric_stats, categorical_stats, 
                    column_selector, column_selector, eda_plots_output, 
                    model_metrics, model_plot, feature_imp_plot, model_select]
        )
        
        def update_column_selector(file):
            if not file:
                return []
            columns = pd.read_csv(file.name).columns.tolist()
            return columns
            
        file_input.change(
            update_column_selector,
            inputs=[file_input],
            outputs=[column_selector]
        )
        
        generate_plot_btn.click(
            lambda plot_type, columns, file: create_custom_plot(plot_type, columns, None, file),
            inputs=[plot_type, column_selector, file_input],
            outputs=[eda_plots_output]
        )
        
        download_btn.click(
            download_model,
            inputs=[model_select],
            outputs=[download_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
