import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from .logger import get_logger


def plot_binarisation_choice(out_cohesion):
  logger = get_logger()
  fig = px.scatter(
    out_cohesion,
    x='pair',
    y='Labels',
    color='Binary Labels',
    labels={'X': 'X-Axis Label', 'Y': 'Y-Axis Label'},
    color_continuous_scale='bluered_r',
    title='Anerage score distribution per pair and binary selection threshold'
  )


  fig.add_shape(
    type='line',
    x0=0, x1=44,
    y0=4.5, y1=4.5,
    line=dict(color='blue', width=2, dash='dash'),
  )


  fig.add_annotation(
    x=22,
    y=4.7,
    text='Threshold: 4.5',
    showarrow=False,
    font=dict(size=12, color='black'),
    align='center'
  )
  fig.show()

def label_distribution(data):
  logger = get_logger()
  label_counts = data.value_counts()
  if 'Labels' in data.columns:
    label_counts = data['Labels'].value_counts()

    # Create a Plotly pie chart
    fig = px.pie(
        names=label_counts.index,
        values=label_counts.values,
        title="Distribution of binary Labels for questionnaire results. The threshold is 4.5 (for ranks 1-6)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.show()
  else:
    logger.info("Column 'label' not found in the CSV file. Please check the file structure.")

def plot_heatmap(heatmap_data):
  logger = get_logger()
  plt.figure(figsize=(8, 6))
  sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='viridis')
  plt.title('F1 Score Heatmap')
  plt.xlabel('Gamma')
  plt.ylabel('Kernel type')
  plt.show()

def plot_grid_search_results(results_df):
  
  logger = get_logger()

  if 'mean_test_f1' in results_df.columns:
    plot_data = pd.DataFrame({
        'Index': range(len(results_df['mean_test_f1'])),
        'F1': results_df['mean_test_f1']
    })

    # Find the best F1 score
    best_f1 = plot_data['F1'].max()

    fig = px.line(
        plot_data,
        x='Index',
        y='F1',
        title='F1 Score Evolution During Grid Search',
        labels={'Index': 'Parameter Combination Index', 'F1': 'F1 Score'},
        markers=True
    )

    fig.add_shape(
        type="line",
        x0=plot_data['Index'].min(),
        x1=plot_data['Index'].max(),
        y0=best_f1,
        y1=best_f1,
        line=dict(color="red", width=2, dash="dash"),
        name="Best F1 Score"
    )

    fig.add_annotation(
        x=plot_data['Index'].max() - 1,
        y=best_f1 + 0.02,
        text=f"Best F1 Score: {best_f1:.3f}",
        showarrow=False,
        font=dict(size=12, color="red"),
        align="right"
    )

    fig.update_layout(
        margin=dict(r=120),
        title=dict(x=0.5)
    )
    fig.show()
    return plot_data
  else:
    logger.info("Column 'mean_test_f1' not found in the DataFrame. Please check the column name.")

def grid_search_trend(plot_data):
  fig = px.scatter(plot_data,
        x='Index',
        y='F1',
        trendline='ols',
        title='F1 Score Trend Visualization',
        labels={'Index': 'Parameter Combination Index', 'F1': 'F1 Score'})
  
  fig.data[1].line.color = "green"
  
  fig.show()

def plot_f1_scores(dataset_scores):
  logger = get_logger()
  fig = px.bar(
      dataset_scores,
      x='Best F1 Score',
      y='Dataset',
      orientation='h',
      color='Dataset',
      text='Best F1 Score',
      title="Best F1 Score by Dataset"
  )

  fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
  fig.update_layout(
      xaxis=dict(title='Best F1 Score', range=[0.7, dataset_scores['Best F1 Score'].max() + 0.02]),
      yaxis_title='Dataset',
      showlegend=False,
      height=1.2 * len(dataset_scores['Dataset'].unique()) * 100
  )

  fig.show()

def plot_pca(components, features, n_features=15):

    logger = get_logger()
    for i in range(len(components)):
        component = components[i]
        top_indices = np.argsort(np.abs(component))[-n_features:]  # Top features by absolute weight
        top_features = features[top_indices]
        top_weights = component[top_indices]

        # Create a horizontal bar chart using Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_weights,
            y=top_features,
            orientation='h',
            marker=dict(color='yellowgreen', opacity=0.8),
            name=f'PC{i+1}'
        ))

        # Customize the layout
        fig.update_layout(
            title=f'Top {n_features} Features in PC{i+1}',
            xaxis_title='Weight',
            yaxis_title='Features',
            yaxis=dict(autorange='reversed'),  # Reverse the y-axis for barh-like behavior
            template='plotly_white',
            height=500
        )

        fig.show()

def plot_cross_validation_results(fold, train_losses, val_losses, train_accs, val_accs):

    logger = get_logger()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Fold {fold+1} Loss", f"Fold {fold+1} Accuracy"]
    )

    epochs_range = list(range(1, len(train_losses)+1))

    fig.add_trace(
        go.Scatter(x=epochs_range, y=train_losses, mode='lines+markers', name='Train Loss'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs_range, y=val_losses, mode='lines+markers', name='Val Loss'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs_range, y=train_accs, mode='lines+markers', name='Train Acc'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs_range, y=val_accs, mode='lines+markers', name='Val Acc'),
        row=1, col=2
    )

    fig.update_layout(
        height=500, width=1000,
        title_text=f"Results for Fold {fold+1}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        xaxis2_title="Epoch",
        yaxis2_title="Accuracy",
        legend_title="Legend",
        showlegend=True
    )

    fig.show()
