import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC



def plot_colorbar(ax, colors, square_len=20):
    """ Turns set of colors into an image with one square per color
    """
    image = np.zeros((square_len, square_len*colors.shape[0], 3))
    for i in range(colors.shape[0]):
        image[:, i*square_len:(i+1)*square_len, :] = colors[np.newaxis, i, :]
    ax.imshow((255.0 * image).astype(np.uint8))



def generate_colors_rgb(n):
    """ Creates equi-distant grid in 3D RGB space
    Number of colors generated is n**3.
    """
    color = np.linspace(0.0, 1.0, n)
    c0, c1, c2 = np.meshgrid(color, color, color, indexing='ij')
    colors = np.vstack([c0.ravel(), c1.ravel(), c2.ravel()]).T
    return colors



def generate_colors(verbose=False):
    """ Uses matplotlib color map to generate fixed number of colors
    """
    tab20b = plt.get_cmap('tab20b')
    tab20c = plt.get_cmap('tab20c')
    steps = np.linspace(0.0, 1.0, 20)
    colors = np.vstack((tab20b(steps), tab20c(steps)))[:,0:3]
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_colorbar(ax, colors)
    return colors



def match_colors(colors, model_colors, verbose=False):
    assert colors.ndim == 2
    assert colors.shape[1] == 3
    assert model_colors.ndim == 2
    assert model_colors.shape[1] == 3
    # Train model using support vector machine
    model = SVC(kernel='linear', C=1e10)
    model_indices = np.arange(model_colors.shape[0])
    model.fit(model_colors, model_indices)
    # Fit data
    predict_indices = model.predict(colors)
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        plot_colorbar(ax, colors)
        ax = fig.add_subplot(212)
        plot_colorbar(ax, model_colors[predict_indices])
    return predict_indices
