import numpy as np
import pylab as pl
def select_region(data, aperiod, bperiod, cell):
    size = data.shape
    a_size = size[0]
    b_size = size[1]

    a_size_min = 0
    a_size_max = int(aperiod * a_size)
    b_size_min = 0
    b_size_max = int(bperiod * b_size)

    a_new_size = a_size_max - a_size_min
    b_new_size = b_size_max - b_size_min

    a = np.zeros((a_new_size, b_new_size))
    b = np.zeros((a_new_size, b_new_size))
    data_new = np.zeros((a_new_size, b_new_size))

    c_i = 0
    for i in range(a_size_min, a_size_max):
        c_j = 0
        for j in range(b_size_min, b_size_max):
            coordinate = cell[0] * i / a_size + cell[1] * j / b_size
            a[c_i][c_j] = coordinate[0]
            b[c_i][c_j] = coordinate[1]
            data_new[c_i][c_j] = data[i % a_size][j % b_size]
            c_j += 1
        c_i += 1
    return data_new, a, b


def contour_plot(data, aperiod, bperiod, cell, figure_name=0, mode='auto', vmin=0, vmax=1):
    new_data, X, Y = select_region(data=data, aperiod=aperiod, bperiod=bperiod, cell=cell)

    fig = pl.figure(figure_name)
    ax = fig.add_subplot(111, adjustable='box', aspect=1)
    level = np.linspace(0, 5, 100)

    if mode == 'auto':
        fig.colorbar(ax.contourf(X, Y, new_data, 200, level=level))
    elif mode == 'absolute':
        fig.colorbar(ax.contourf(X, Y, new_data, 200, level=level, vmin=vmin, vmax=vmax))

    return fig, ax
