import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def get_data():
    dataset_path = os.path.join(os.path.dirname(__file__), 'Data.csv')
    dataset = pd.read_csv(dataset_path)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return x, y


def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=0)


def visualize(y_test, y_predicted, name):
    plot_points_count = 25

    plot_x = np.arange(0, plot_points_count)
    plot_y_test = y_test[:plot_points_count]
    plot_y_predicted = y_predicted[:plot_points_count]

    r2 = r2_score(y_test, y_predicted)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_title(name)

    ax.plot(plot_x, plot_y_test, color='green', label='Test')
    ax.plot(plot_x, plot_y_predicted, color='red', label='Predicted')

    ax.text(0.95, 0.95, f'R² = {r2:.4f}',
            verticalalignment='top', horizontalalignment='right', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat'))

    ax.legend()
    plt.show()
