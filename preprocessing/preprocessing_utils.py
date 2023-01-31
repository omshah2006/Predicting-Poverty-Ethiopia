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
    
    BAND_MEANS = {'BLUE': 0.05720699718743952, 'GREEN': 0.09490949383988444, 'RED': 0.11647556706520566, 'NIR': 0.25043694995276194, 'SW_IR1': 0.2392968657712096, 'SW_IR2': 0.17881930908670116, 'TEMP': 309.4823962960872, 'avg_rad': 1.8277193893627437}
    BAND_STDS = {'BLUE': 0.02379879403788589, 'GREEN': 0.03264212296594092, 'RED': 0.050468921297598834, 'NIR': 0.04951648377311826, 'SW_IR1': 0.07332469136800321, 'SW_IR2': 0.07090649886221509, 'TEMP': 6.000001012494749, 'avg_rad': 4.328436715534132}

    
    r, g, b = bands['RED'], bands['GREEN'], bands['BLUE']
#     r, g, b = ((bands['RED'] - BAND_MEANS['RED']) / BAND_STDS['RED']) , ((bands['GREEN'] - BAND_MEANS['GREEN']) / BAND_STDS['GREEN']), ((bands['BLUE'] - BAND_MEANS['BLUE']) / BAND_STDS['BLUE'])
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