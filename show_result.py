
import matplotlib.pyplot as plt
import numpy as np



def plot_result(label_text, x, list_y, list_y_filtered, raw_d1, filtered_d1):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(x, list_y)
    ax[0, 0].set_title(label_text + ":::raw y")
    ax[0, 1].plot(x, list_y_filtered[0])
    ax[0, 1].set_title(label_text + ":::filtered y")
    ax[1, 0].plot(x, raw_d1)
    ax[1, 0].set_title(label_text + ":::raw d1")
    ax[1, 1].plot(x, filtered_d1)
    ax[1, 1].set_title(label_text + ":::filtered d1")
    plt.show()


def save_result(label_text, x, list_y, list_y_filtered, raw_d1, filtered_d1):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(x, list_y)
    ax[0, 0].set_title(label_text + ":::raw y")
    ax[0, 1].plot(x, list_y_filtered[0])
    ax[0, 1].set_title(label_text + ":::filtered y")
    ax[1, 0].plot(x, raw_d1)
    ax[1, 0].set_title(label_text + ":::raw d1")
    ax[1, 1].plot(x, filtered_d1)
    ax[1, 1].set_title(label_text + ":::filtered d1")
    plt.savefig(label_text + '.png')
