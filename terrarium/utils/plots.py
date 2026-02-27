'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Video
from matplotlib import colormaps
from tqdm import tqdm
from ._utils import to_list
from .sugar import attrmethod

__all__ = [
    'grab_plot',
    'show_data',
    'VideoWriter'
]

def _format_ax(ax, title=None, show_ax=True):
    title and ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    not show_ax and ax.axis('off')

    return ax

def _make_plot(n_plots=1, fig_size=5, colormap=None):
    fig_size = to_list(2, n_plots*fig_size)
    fig, ax = plt.subplots(1, n_plots, figsize=fig_size)
    (n_plots == 1) and (ax := (ax,))

    if colormap is not None:
        cmap = colormaps.get_cmap(colormap)

        return fig, ax, cmap
    return fig, ax

def _show_plot(data, plot, ax, title, data_idx=6, split_by='Day', show_ax=True):
    mask = data[split_by] == plot
    p_data = data[mask].to_numpy()[:, data_idx:]
    p_data = len(p_data)*(p_data/p_data.sum(1)[:, None])
    x_range = np.arange(len(p_data))
    ax.bar(x_range, p_data[:, 0])

    for i in range(1, p_data.shape[1]):
        ax.bar(x_range, p_data[:, i], bottom=p_data[:, :i].sum(1))

    _format_ax(ax, title, show_ax)

def show_data(data, data_idx=None, split_by='Day', fig_size=5, colormap='Set3', show_ax=False, title=None, path=None, return_plot=False, verbosity=1):
    data_idx is None and (data_idx := np.argmax([True if k.isupper() and not k.isalpha() else False for k in data.keys()]))
    n_plots = len(plots := np.unique(data[split_by]))
    title = to_list(n_plots, title) if title is not None else [f'{split_by} {x}' for x in plots]
    fig, ax, cmap = _make_plot(n_plots, fig_size, colormap)

    for p, a, t in tqdm(zip(plots, ax, title), total=n_plots) if verbosity == 1 else zip(plots, ax, title):
        _show_plot(data, p, a, t, data_idx, split_by, show_ax)

    fig.tight_layout()
    path and fig.savefig(path, bbox_inches='tight', transparent=True)

    if return_plot:
        return fig, ax

    plt.show()

def grab_plot(close=True, return_tensor=False):
    (fig := plt.gcf()).canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    alpha = np.float32(img[..., 3:]*255.)
    img = np.uint8(255.*(1. - alpha) + img[..., :3]*alpha)

    if close:
        plt.close()

    if return_tensor:
        return torch.tensor(img)
    return img

class VideoWriter:
    @attrmethod
    def __init__(self, frame_size=500, frame_rate=30., path='_autoplay.mp4'):
        self.frames = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.save()

        if self.path[-13:] == '_autoplay.mp4':
            self.show()

    def write(self, frame):
        self.frames.append(frame)

    def save(self):
        with imageio.imopen(self.path, 'w', plugin='pyav') as out:
            out.init_video_stream('vp9', fps=self.frame_rate)

            for frame in self.frames:
                out.write_frame(frame)

    def show(self):
        out = Video(self.path, width=self.frame_size, height=self.frame_size)
        display(out)
