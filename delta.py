def Lorentzian(value, width):
    import numpy as np

    K = width / np.pi
    return K / (value**2 + width**2)

def Gaussian(value, width):
    import numpy as np

    K = 1 / ((2 * np.pi)**(1/2)*width) * np.exp(-value**2/(2 * width**2))
    return K