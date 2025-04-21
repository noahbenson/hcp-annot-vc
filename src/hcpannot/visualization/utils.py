import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def save_fig(save_path):
    if save_path is not None:
        parent_path = Path(save_path)
        if not os.path.exists(parent_path.parent.absolute()):
            os.makedirs(parent_path.parent.absolute())
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

def remove_all_ticks_from_fig(axes):
    for ax in axes.flatten():
        ax.set_yticks([])            # Remove y-axis ticks
        ax.set_yticklabels([])        # Remove y-axis tick labels
        ax.set_xticks([])            # Remove x-axis ticks
        ax.set_xticklabels([])        # Remove x-axis tick labels
 
def set_rcParams(rc):
    if rc is None:
        pass
    else:
        for k, v in rc.items():
            plt.rcParams[k] = v
            
            
def get_height_based_on_width(width, aspect_ratio):

    return aspect_ratio * width
            