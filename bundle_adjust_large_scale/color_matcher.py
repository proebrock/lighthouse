import numpy as np
import matplotlib.pyplot as plt



def generate_colors_rgb(p):
    color = np.linspace(0.0, 1.0, p)
    c0, c1, c2 = np.meshgrid(color, color, color, indexing='ij')
    colors = np.vstack([c0.ravel(), c1.ravel(), c2.ravel()]).T
    return colors

def generate_colors():
    tab20b = plt.get_cmap('tab20b')
    tab20c = plt.get_cmap('tab20c')
    steps = np.linspace(0.0, 1.0, 20)
    return np.vstack((tab20b(steps), tab20c(steps)))[:,0:3]

def plot_colorbar(colors, square_len=20):
    image = np.zeros((square_len, square_len*colors.shape[0], 3))
    for i in range(colors.shape[0]):
        image[:, i*square_len:(i+1)*square_len, :] = colors[np.newaxis, i, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow((255.0 * image).astype(np.uint8))
    plt.show()
