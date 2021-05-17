from gpaw import GPAW
import numpy as np
import delta
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
from scipy.signal import find_peaks

counter = 0


def fun(x, A, k):
    return A - k * x

def isopoint(a, isovalue, accuracy=1e-3):
    global counter

    from scipy.interpolate import interp1d
    a = np.real(a)
    index = len(a) - 1
    for i in range(0, len(a)):
        index = len(a) - i - 1
        if a[index] - isovalue >= 1e-15:
            break
    if index == len(a) - 1:
        print('AAA')
        return 0
    i_min = index / (len(a) - 1)
    i_max = (index + 1) / (len(a) - 1)
    xp = np.linspace(0, 1, len(a))
    f = interp1d(xp, a, kind=3)
    counter += 1

    h = find_value(isovalue=isovalue, fun=f, x_min=i_min, x_max=i_max, accuracy=accuracy)

    return h

def find_value(isovalue, fun, x_min, x_max, accuracy=1e-3):
    f_max = float(fun(x_min))
    f_min = float(fun(x_max))

    x_mid = (x_min + x_max) / 2
    f_mid = float(fun(x_mid))

    if not (f_min <= isovalue and isovalue <= f_max):
        print('Error: Loss')
        print(counter)
        return 0

    if abs(f_min - f_max) <= accuracy:
        return x_mid

    elif isovalue >= f_mid:
        return find_value(isovalue, fun, x_min, x_mid, accuracy=accuracy)
    else:
        return find_value(isovalue, fun, x_mid, x_max, accuracy=accuracy)


'''
x = [1, 0.5, 1/3, 1/4, 1/5, 1/6]

print(isopoint(x, 0.2760465, accuracy=1e-4))
'''





class NAVI:

    '''
    Source Documents
    '''
    atom = None
    calc = None
    cell = None

    '''
    Energy Information
    '''
    potential_energy = 0
    ef = 0
    eigenvalue = None

    '''
    k Space Information
    '''
    nband = 0
    nikpts = 0
    nkpts = 0
    spin_polarized = 0
    ikpts = None
    kpts = None
    kpts_weight = None
    kpte_map = None

    '''
    Surface Information
    '''
    height = 0
    surface_height = 0
    zmin = 0
    zmax = 0

    '''
    Plot Infromation
    '''
    aperiod = 1
    bperiod = 1
    __figure_count = 0

    '''
    Supporting Functions
    '''
    width = 0.026
    delta_function = None
    fitting_function = None

    '''
    Free Space
    '''
    free_space_data = None


    def __init__(self, file = None):

        print('Welcome to NAVI System!                          \n'
              '                 o      o                        \n'
              '                o o    o o                       \n'
              '               o   o  o   o                      \n'
              '              o  o o o o o o                     \n'
              '            o                o                   \n'
              '           o      /\     /\    o                 \n'
              '           o                   o                 \n'
              '            o    @    \_/  @  o                  \n'
              '               o             o                   \n'
              '                  o o o o o                      \n'
              )
        print('Loading Data...')
        if file:
            try:
                self.calc = GPAW(restart=file)
            except FileNotFoundError:
                raise FileNotFoundError('Fail to open gpw file: ' + file)
            except TypeError:
                raise TypeError('File name must be a string.')
            else:
                pass

            self.atoms = self.calc.get_atoms()
            self.cell = self.atoms.get_cell()

            self.potential_energy = self.calc.get_potential_energy()
            self.ef = self.calc.get_fermi_level()

            self.nband = self.calc.get_number_of_bands()
            self.spin_polarized = self.calc.get_spin_polarized()
            self.kpts = self.calc.get_bz_k_points()
            self.ikpts = self.calc.get_ibz_k_points()
            self.nkpts = len(self.kpts)
            self.nikpts = len(self.ikpts)
            self.kpts_weight = self.calc.get_k_point_weights()
            self.kpts_map = self.calc.get_bz_to_ibz_map()

            self.height = self.cell[2][2]
            self.surface_height = max([u[2] for u in self.atoms.get_positions()])
            self.zmin = 0
            self.zmax = self.height

            if self.spin_polarized:
                self.eigenvalue = np.zeros((self.nikpts, self.nband, 2))
            else:
                self.eigenvalue = np.zeros((self.nikpts, self.nband, 1))
            if self.spin_polarized:
                spin = 2
            else:
                spin = 1
            for i in range(0, self.nikpts):
                for k in range(0, spin):
                    eig = self.calc.get_eigenvalues(kpt=i, spin=k)
                    for j in range(0, self.nband):
                        self.eigenvalue[i][j][k] = eig[j]

            self.delta_function = delta.Lorentzian
            self.fitting_function = fun

        print('Data loaded sucessfully!')

    def create_probability_density_file(self, prefix='./Density', energy_range=3.5):
        dir = prefix + '/'
        if not os.path.exists(dir):
            os.mkdir(dir)

        if self.spin_polarized:
            spin = 2
        else:
            spin = 1

        count = 1
        print_count = 0
        dictionary = np.zeros((self.nikpts, self.nband, 2))
        print('Create Density File: ')
        N_tot = self.nikpts * self.nband * spin

        for i in range(0, self.nikpts):
            for j in range(0, self.nband):
                for k in range(0, spin):
                    if abs(self.ef - self.eigenvalue[i][j][k]) > energy_range:
                        print_count += 1
                        perc = print_count / N_tot * 100
                        print('%.4f %%\r' % perc, end='')
                        continue

                    wf = self.calc.get_pseudo_wave_function(kpt=i, band=j, spin=k)
                    wf_dens = np.real(wf * np.conj(wf))
                    np.save(dir + str(count) + '.npy', wf_dens)
                    dictionary[i][j][k] = count
                    count += 1

                    print_count += 1
                    perc= print_count/N_tot * 100
                    print('%.4f %%\r' % perc, end='')

        np.save(dir + 'dictionary', dictionary)
        return None

    def ldos(self, V=0.0, read_dir='./Density', free_space=False):
        LDOS = np.zeros(self.calc.get_pseudo_wave_function().shape)

        if self.spin_polarized:
            spin = 2
        else:
            spin = 1

        dir = read_dir + '/'
        dictionary = np.load(dir + 'dictionary.npy')

        for i in range(0, self.nikpts):
            for j in range(0, self.nband):
                for k in range(0, spin):

                    if dictionary[i][j][k] == 0:
                        continue
                    else:
                        file = str(int(dictionary[i][j][k])) + '.npy'

                    dE = self.eigenvalue[i][j][k] - self.ef - V
                    wf_dens = np.load(dir + file)
                    delta = self.delta_function(value=dE, width=self.width)
                    LDOS += wf_dens * delta * self.kpts_weight[i] * self.nkpts

        return LDOS


    def I_t(self, V=0.0, read_dir='./Density', e=0.2):
        print('#######################################')
        print('#Tunneling current calculation module.#')
        print('#######################################')
        print('#Calculate Tunneling Current at V = ' + str(V))
        I = np.zeros(self.calc.get_pseudo_wave_function().shape)
        N = abs(int(V / e))
        if V >= 0:
            value = np.linspace(0, V, N)
        else:
            value = np.linspace(V, 0, N)
        for u in value:
            I += self.ldos(V=u, read_dir=read_dir) * abs(V) / N
            print('LDOS at ' + str(u) +'\r', end='')

        self.free_space_data = I

        np.save(file='I.npy', arr=I)

        return I

    def isosurface_scanning(self, data, fixed_value=1e-4, type='fitting'):
        size = data.shape
        size_a = size[0]
        size_b = size[1]
        size_c = size[2]

        Nz_min = int(self.zmin / self.height * size_c)
        Nz_max = int(self.zmax / self.height * size_c)

        real_zmin = Nz_min / size_c * self.height
        real_zmax = Nz_max / size_c * self.height

        def fitting():
            h = np.linspace(real_zmin - self.surface_height, real_zmax - self.surface_height, Nz_max - Nz_min)
            H = np.zeros((size_a, size_b))
            for i in range(0, size_a):
                for j in range(0, size_b):
                    y = data[i, j, Nz_min:Nz_max]
                    # pl.plot(h, y)
                    # pl.show()
                    # y[y <= 1e-12] = 1e-12
                    y = np.log(y)
                    result, error = curve_fit(f=fun, xdata=h, ydata=y)
                    H[i][j] = (result[0] - np.log(fixed_value)) / result[1]

            return H

        def interpolate():
            H = np.zeros((size_a, size_b))
            print_count = 0
            N_tot = size_a * size_b
            for i in range(0, size_a):
                for j in range(0, size_b):
                    y = data[i, j, Nz_min:Nz_max]
                    e = isopoint(a=y, isovalue=fixed_value, accuracy=1e-8)
                    h = e * (real_zmax - real_zmin) + self.surface_height - real_zmin
                    H[i][j] = h

                    print_count += 1
                    perc= print_count/N_tot * 100
                    print('%.4f %%\r' % perc, end='')



            return H

        if type == 'fitting':
            return fitting()

        if type == 'interpolate':
            return interpolate()


    def isoheight_scanning(self, data):
        pass


    def select_region(self, data):
        size = data.shape
        a_size = size[0]
        b_size = size[1]

        a_size_min = 0
        a_size_max = int(self.aperiod * a_size)
        b_size_min = 0
        b_size_max = int(self.bperiod * b_size)

        a_new_size = a_size_max - a_size_min
        b_new_size = b_size_max - b_size_min

        a = np.zeros((a_new_size, b_new_size))
        b = np.zeros((a_new_size, b_new_size))
        data_new = np.zeros((a_new_size, b_new_size))

        c_i = 0
        for i in range(a_size_min, a_size_max):
            c_j = 0
            for j in range(b_size_min, b_size_max):
                coordinate = self.cell[0] * i / a_size + self.cell[1] * j / b_size
                a[c_i][c_j] = coordinate[0]
                b[c_i][c_j] = coordinate[1]
                data_new[c_i][c_j] = data[i % a_size][j % b_size]

                c_j += 1
            c_i += 1
        return data_new, a, b

    def contour_plot(self, data):
        new_data, X, Y = self.select_region(data)

        fig = pl.figure(self.__figure_count)
        ax = fig.add_subplot(111, adjustable='box', aspect=1)
        self.__figure_count += 1

        fig.colorbar(ax.contourf(X, Y, new_data, 100, cmap='gray'))

        center = 1 / 2 * self.cell[0] + 1 / 2 * self.cell[1]
        ax.scatter(center[0], center[1], color='r')
        return fig, ax

    def fermi_level_reculate(self, N):
        return self.__devide_find(N, E_min=-10, E_max=10)

    def __calculate_occupation_number(self, Ef):
        if self.spin_polarized:
            spin = 2
            occupation = 1
        else:
            spin = 1
            occupation = 2

        N_calc = 0
        for i in range(0, self.nikpts):
            n_k = self.kpts_weight[i] * self.nkpts
            for j in range(0, self.nband):
                for k in range(0, spin):
                    E = self.eigenvalue[i][j][k]
                    N_calc += 1 / (1 + np.exp((E - Ef)/self.width)) * n_k * occupation

        return N_calc

    def __devide_find(self, N, E_min, E_max):
        E_try = (E_min + E_max) / 2
        N_try = self.__calculate_occupation_number(E_try)
        if abs(N_try - N) <= 0.1:
            return E_try
        elif N_try > N:
            return self.__devide_find(N, E_min=E_min, E_max=E_try)
        else:
            return self.__devide_find(N, E_min=E_try, E_max=E_max)


    def PLOT_STM_IMAGE(self, read_dir='./Density', V=0.0, I_fix=1e-4, e=0.2, type='fitting', restart=None):
        print('#############################')
        print('Get Task: Plot STM Image')
        print('Check the configuration.')
        if restart:
            print('Input file exist, Loading Data...')
            I = np.load(file='I.npy')
            print('Data loaded successfully!')
            A.free_space_data = I
        else:
            print('No input file, Construct tunneling current data')
            I = self.I_t(V=V, e=e, read_dir=read_dir)
        H = self.isosurface_scanning(data=I, fixed_value=I_fix, type=type)
        fig, ax = self.contour_plot(H)
        return fig, ax

    def PLOT_ISO_IMAGE(self, data, fixed_value, type='fitting', a_min=0.0, a_max=10.0):
        H = self.isosurface_scanning(data=data, fixed_value=fixed_value, type=type)
        fig, ax = self.contour_plot(np.clip(H, a_min=a_min, a_max=a_max))
        return fig, ax

if __name__ == '__main__':
    A = NAVI(file='AU_Nb_6_A_U6.0_recalculate.gpw')
    A.zmin = A.surface_height
    A.zmax = A.zmax - 1.5
    A.aperiod = 3
    A.bperiod = 3

    '''
    E, DOS = A.calc.get_dos(npts=4000)
    E = E - A.ef
    pl.plot(E, DOS)
    pl.show()
    '''

    '''
    e_Nb_d, ldos_Nb_d = A.calc.get_orbital_ldos(a=0, angular='d', npts=5000, spin=0, width=A.width * 5)
    f = pl.figure(1)
    ax = f.add_subplot(111)
    

    ax.plot(e_Nb_d - A.ef, ldos_Nb_d)
    pl.show()
    f.savefig('A.png')

    calc = A.calc

    E, DOS = calc.get_dos(npts=4000)
    E = E - A.ef
    pl.plot(E, DOS)
    pl.show()
    '''
    #A.create_probability_density_file(energy_range=2.5)
    '''
    fig, ax = A.PLOT_STM_IMAGE(V=-1.0,
                               e=0.1,
                               I_fix=0.03,
                               type='interpolate',
                               #restart='I.npy'
                               )
    pl.show()
    while True:
        I = input('Change I:')
        if I == 'exit':
            break
        fig2, ax2 = A.PLOT_ISO_IMAGE(data=A.free_space_data,
                                     fixed_value=float(I),
                                     type='interpolate')
        pl.show()


   '''


    data = np.load('I_1.5.npy')
    '''
    data2 = np.zeros(data.shape)
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[2]):
                data2[i, j, k] = data[i, j, data.shape[2] - k - 1]
    '''
    fig2, ax2 = A.PLOT_ISO_IMAGE(data=data,
                                 fixed_value=2.9e-2,
                                 type='interpolate')
    pl.show()


    while True:
        I = input('Change I:')
        a_min = input('a_min')
        a_max = input('a_max')
        if I == 'exit':
            break
        fig2, ax2 = A.PLOT_ISO_IMAGE(data=data,
                                     fixed_value=float(I),
                                     type='interpolate',
                                     a_min=float(a_min),
                                     a_max=float(a_max)
                                     )
        pl.show()




