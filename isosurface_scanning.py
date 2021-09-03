import numpy as np
import fitting_function
from scipy.optimize import curve_fit

def isopoint(a, isovalue, accuracy=1e-5):
    from scipy.interpolate import interp1d
    a = np.real(a)
    index = len(a) - 1
    for i in range(0, len(a)):
        index = len(a) - i - 1
        if a[index] - isovalue >= 1e-5 * accuracy:
            break
    if index == len(a) - 1:
        print('Error: Cannot Fine Coarse Grid with Isovalue')
        return 0
    i_min = index / (len(a) - 1)
    i_max = (index + 1) / (len(a) - 1)
    xp = np.linspace(0, 1, len(a))
    f = interp1d(xp, a, kind=3)

    h = find_value(isovalue=isovalue, fun=f, x_min=i_min, x_max=i_max, accuracy=accuracy)
    return h

def find_value(isovalue, fun, x_min, x_max, accuracy=1e-5):
    f_max = float(fun(x_min))
    f_min = float(fun(x_max))

    x_mid = (x_min + x_max) / 2
    f_mid = float(fun(x_mid))

    if not (f_min <= isovalue and isovalue <= f_max):
        print('Error: Loss')
        return 0

    if abs(f_min - f_max) <= accuracy:
        return x_mid

    elif isovalue >= f_mid:
        return find_value(isovalue, fun, x_min, x_mid, accuracy=accuracy)
    else:
        return find_value(isovalue, fun, x_mid, x_max, accuracy=accuracy)



class iso_scanner:
    __slots__ = {'scanner_type', 'accuracy', 'isovalue'}
    __default = {'scanner_type': 'interpolate', 'accuracy':1e-5, 'isovalue':0.01}

    def __init__(self, scanner_type=None, accuracy=None, isovalue=None):
        if scanner_type == None:
            self.scanner_type = self.__default['scanner_type']
        else:
            self.scanner_type = scanner_type

        if accuracy == None:
            self.accuracy = self.__default['accuracy']
        else:
            self.accuracy = accuracy

        if isovalue == None:
            self.isovalue = self.__default['isovalue']
        else:
            self.isovalue = isovalue

    def scanning(self, data, Nz_min, Nz_max, h_min=0.0, h_max=1.0):
        size = data.shape
        size_a = size[0]
        size_b = size[1]
        size_c = size[2]

        def fitting_scanning():
            h = np.linspace(h_min, h_max, Nz_max - Nz_min)
            H = np.zeros((size_a, size_b))
            for i in range(0, size_a):
                for j in range(0, size_b):
                    y = data[i, j, Nz_min:Nz_max]
                    y = np.log(y)
                    result, error = curve_fit(f=fitting_function.linear, xdata=h, ydata=y)
                    H[i][j] = (result[0] - np.log(self.isovalue)) / result[1]
            return H

        def interpolate_scanning():
            H = np.zeros((size_a, size_b))
            print_count = 0
            N_tot = size_a * size_b
            for i in range(0, size_a):
                for j in range(0, size_b):
                    y = data[i, j, Nz_min:Nz_max]
                    e = isopoint(a=y, isovalue=self.isovalue, accuracy=self.accuracy)
                    h = e * (h_max - h_min) - h_min
                    H[i][j] = h
                    print_count += 1
                    perc = print_count / N_tot * 100
                    print('%.4f %%\r' % perc, end='')
            return H

        if self.scanner_type == 'fitting':
            return fitting_scanning()

        if self.scanner_type == 'interpolate':
            return interpolate_scanning()
