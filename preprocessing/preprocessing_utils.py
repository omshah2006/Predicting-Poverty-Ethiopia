import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def get_feature_array(feature_set):
    result = {}
    for key, feature in feature_set.items():
        kind = feature.WhichOneof('kind')
        shape = np.array(getattr(feature, kind).value).shape
        result[key] = kind, shape
       
    return result

def create_single_feature_set(filename):
    record = tf.data.TFRecordDataset(filenames=[filename])
    feature_set = parse_features(record=record)
    
    return feature_set

def parse_features(record):
    raw_example = next(iter(record)) 
    example = tf.train.Example.FromString(raw_example.numpy())
    
    return example.features.feature

def visualize_bands(img, band_names, bands_per_row=3):
    n_rows = math.ceil(len(band_names) / bands_per_row)
    fig, axs = plt.subplots(nrows=n_rows, ncols=bands_per_row, sharex=True, sharey=True)
    
    bands = {band_name: img[:, :, b] for b, band_name in enumerate(band_names)}
    
    plots = []
    plot_names = []
    
    r, g, b = bands['RED'], bands['GREEN'], bands['BLUE']
    rgb = np.stack([r, g, b], axis=2)
    plots.append(rgb)
    plot_names.append('RGB')
    
    for band_name in band_names:
        plots.append(bands[band_name])
        plot_names.append(band_name)
    
    for b in range(len(plots)):
        if len(axs.shape) == 1:
            ax = axs[b]
        else:
            ax = axs[b // bands_per_row, b % bands_per_row]
        # set origin='lower' to match lat/lon direction
        im = ax.imshow(plots[b], origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(plot_names[b])
    
    fig.set_size_inches(10, 8, forward=True)
    fig.colorbar(im, orientation='vertical', ax=axs)
        
    plt.show()