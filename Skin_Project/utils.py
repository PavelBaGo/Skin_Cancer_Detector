import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ml_logic.model import evaluate_model



def plot_metrics_threshold(model,X_test,y_test):
    threshold = []
    for i in np.arange(0.1,0.5,0.03):
        threshold.append(i)
    df_results = pd.DataFrame()
    for i in threshold:
        df_results = pd.concat([evaluate_model(model,X_test,y_test,i),df_results.loc[:]]).reset_index(drop=True)

    plt.figure(figsize=(10,6))
    plt.plot(df_results['Threshold'],df_results['Accuracy'],label='Accuracy')
    plt.plot(df_results['Threshold'],df_results['Precision'],label='Precision')
    plt.plot(df_results['Threshold'],df_results['Recall'],label='Recall')
    plt.plot(df_results['Threshold'],df_results['F2 Score'],label='F2 Score')
    plt.title('Metrics evolution with different threshold')
    plt.xlabel("Threshold")
    plt.ylabel("Metrics score")
    plt.legend()
    plt.show()

    pass
