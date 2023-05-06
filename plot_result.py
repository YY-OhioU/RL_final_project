import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

DATA_ROOT = 'training_data'
FILE_FORMAT = 'statics_[0-9]*.pkl'


def gather_file(data_root):
    folder = Path(data_root)
    files = []
    for f in folder.rglob(FILE_FORMAT):
        files.append(f)
    return files


def load_data(file_ordered_list):
    scores = []
    returns = []
    for file in file_ordered_list:
        with open(file, 'rb') as f:
            record = pickle.load(f)
            scores.extend(record[0])
            returns.extend(record[1])
    return scores, returns


def plot_data(data, window_size=50, save_path=None):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Create a new figure and set its size
    fig = plt.figure(figsize=(8, 6))

    # Plot the original data as a line graph
    plt.plot(data, label='Original')

    # Plot the smoothed data as a line graph
    plt.plot(smoothed_data, label=f'Smoothed (over {window_size} episodes)')

    # Set the title and axis labels
    plt.title("Total reward in each episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")

    # Add a legend
    plt.legend()

    # Display the plot
    plt.savefig(save_path)
    plt.show()
    return fig


if __name__ == '__main__':
    fl = gather_file(DATA_ROOT)
    records = load_data(fl)
    window_size = 100
    fig_path = Path(DATA_ROOT) / f'total_reward_smooth{window_size}.png'
    fig = plot_data(records[1], window_size=window_size, save_path=fig_path.resolve())
