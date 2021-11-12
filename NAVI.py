from gpaw import GPAW
import numpy as np
import delta_function
import fitting_function
from isosurface_scanning import iso_scanner
from STM_plot import contour_plot

class NAVI:
    __slots__ = {'atoms', 'calc', 'cell',
                 'potential_energy', 'ef', 'eigenvalue',
                 'nbands', 'nikpts', 'nkpts', 'spin_polarized', 'ikpts', 'kpts', 'kpts_weight', 'kpts_map',
                 'height', 'surface_height', 'zmin', 'zmax', 'shape',
                 'aperiod', 'bperiod', '__figure_count',
                 'width', 'delta_function', 'fitting_function',
                 'figcounter',
                 }
    parameter_dictionary = {'Source Documents': ['atoms', 'calc', 'cell'],
                            'Energy Information': ['potential_energy', 'ef', 'eigenvalue'],
                            'k Space': ['nbands', 'nikpts', 'nkpts', 'spin_polarized', 'ikpts', 'kpts', 'kpts_weight', 'kpts_map'],
                            'Surface': ['height', 'surface_height', 'zmin', 'zmax', 'shape'],
                            'Plot': ['aperiod', 'bperiod', '__figure_count'],
                            'Supporting Functions': ['width', 'delta_function', 'fitting_function']}


    def __init__(self, file=None, mode='all'):

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

            self.nbands = self.calc.get_number_of_bands()
            self.spin_polarized = self.calc.get_spin_polarized()
            self.kpts = self.calc.get_bz_k_points()
            self.ikpts = self.calc.get_ibz_k_points()
            self.nkpts = len(self.kpts)
            self.nikpts = len(self.ikpts)
            self.kpts_weight = self.calc.get_k_point_weights()
            self.kpts_map = self.calc.get_bz_to_ibz_map()
            self.shape = self.calc.get_pseudo_wave_function().shape

            self.height = self.cell[2][2]
            self.surface_height = max([u[2] for u in self.atoms.get_positions()])
            self.zmin = self.surface_height
            self.zmax = self.height

            if self.spin_polarized:
                self.eigenvalue = np.zeros((self.nikpts, self.nbands, 2))
            else:
                self.eigenvalue = np.zeros((self.nikpts, self.nbands, 1))
            if self.spin_polarized:
                spin = 2
            else:
                spin = 1
            for i in range(0, self.nikpts):
                for k in range(0, spin):
                    eig = self.calc.get_eigenvalues(kpt=i, spin=k)
                    for j in range(0, self.nbands):
                        self.eigenvalue[i][j][k] = eig[j]
            self.width = 0.1
            self.delta_function = delta_function.Lorentzian
            self.fitting_function = fitting_function.linear

            if mode == 'all':
                self.shape = self.calc.get_pseudo_wave_function().shape

        self.figcounter = 0
        print('Data loaded sucessfully!')

    def print_parameters(self, type='all'):
        if type == 'all':
            for u in self.parameter_dictionary.keys():
                print(u + ':\n')
                for v in self.parameter_dictionary[u]:
                    print(v + '; ')
        elif type in self.parameter_dictionary.keys():
            print(type + ':\n')
            for u in self.parameter_dictionary[type]:
                print(u + '; ')

    def get_ldos(self, V=0.0, cutting_off=1.0):
        LDOS = np.zeros(self.shape)
        print('LDOS at V=%.3f\n' % V)

        if self.spin_polarized:
            spin = 2
        else:
            spin = 1

        print_count = 0
        N_tot = self.nikpts * self.nbands * spin

        for i in range(0, self.nikpts):
            for j in range(0, self.nbands):
                for k in range(0, spin):
                    if abs((self.eigenvalue[i][j][k] - self.ef) - V) > cutting_off:
                        print_count += 1
                        perc = print_count / N_tot * 100
                        print('%.4f %%\r' % perc, end='')
                    else:
                        dE = self.eigenvalue[i][j][k] - self.ef - V
                        wf = self.calc.get_pseudo_wave_function(band=j, kpt=i, spin=k)
                        wf_dens = np.real(wf * np.conj(wf))
                        #norm = np.sum(wf_dens)
                        delta = self.delta_function(value=dE, width=self.width)
                        #LDOS += wf_dens * delta * self.kpts_weight[i] * self.nkpts
                        f = delta * self.kpts_weight[i] * self.nkpts
                        LDOS += wf_dens * f

                        print_count += 1
                        perc = print_count / N_tot * 100
                        print('%.4f %%\r' % perc, end='')
        return LDOS

    def get_spin_polarized_ldos(self, V=0.0, cutting_off=1.0, spin=0):
        assert self.spin_polarized

        LDOS = np.zeros(self.shape)
        print('LDOS at V=%.3f, Spin=%d\n' % (V, spin))

        print_count = 0
        N_tot = self.nikpts * self.nbands

        for i in range(0, self.nikpts):
            for j in range(0, self.nbands):
                if abs((self.eigenvalue[i][j][spin] - self.ef) - V) > cutting_off:
                    print_count += 1
                    perc = print_count / N_tot * 100
                    print('%.4f %%\r' % perc, end='')
                else:
                    dE = self.eigenvalue[i][j][spin] - self.ef - V
                    wf = self.calc.get_pseudo_wave_function(band=j, kpt=i, spin=spin)
                    wf_dens = np.real(wf * np.conj(wf))
                    #norm = np.sum(wf_dens)
                    delta = self.delta_function(value=dE, width=self.width)
                    # LDOS += wf_dens * delta * self.kpts_weight[i] * self.nkpts
                    f = delta * self.kpts_weight[i] * self.nkpts
                    LDOS += wf_dens * f

                    print_count += 1
                    perc = print_count / N_tot * 100
                    print('%.4f %%\r' % perc, end='')
        return LDOS


    def get_tunneling_current(self, V=0.0, voltage_spacing=0.1, cutting_off=1.0, save=None):
        print('###############################')
        print('#                             #')
        print('#Tunneling Current Calculation#')
        print('#                             #')
        print('###############################')
        print('#Calculate Tunneling Current at V = ' + str(V))
        I = np.zeros(self.shape)
        N = abs(int(V / voltage_spacing))

        if V >= 0:
            value = np.linspace(0, V, N)
        else:
            value = -np.linspace(0, abs(V), N)

        for u in value:
            I += self.get_ldos(V=u, cutting_off=cutting_off) * abs(V) / N
        if save is not None:
            np.save(file=save, arr=I)

        return I

    def get_spin_polarized_tunneling_current(self, V=0.0, voltage_spacing=0.1, cutting_off=1.0, spin=0, save=None):
        print('###############################')
        print('#                             #')
        print('#Tunneling Current Calculation#')
        print('#                             #')
        print('###############################')
        print('#Calculate Tunneling Current at V = ' + str(V))
        I = np.zeros(self.shape)
        N = abs(int(V / voltage_spacing))

        if V >= 0:
            value = np.linspace(0, V, N)
        else:
            value = -np.linspace(0, abs(V), N)

        for u in value:
            I += self.get_spin_polarized_ldos(V=u, cutting_off=cutting_off, spin=spin) * abs(V) / N
        if save is not None:
            np.save(file=save, arr=I)

        return I


    def get_dos(self, e_min, e_max, npts=201, spin=0):
        pre_E, pre_dos = self.calc.get_dos(spin=spin)
        calc_min = pre_E[0]
        calc_max = pre_E[-1]
        tot_npts = int((calc_max - calc_min) / (e_max - e_min) * npts)
        num_min = int((e_min - calc_min)/(calc_max - calc_min) * tot_npts)
        num_max = int((e_max - calc_min)/(calc_max - calc_min) * tot_npts)

        fine_E, fine_dos = self.calc.get_dos(spin=spin, npts=tot_npts)

        return fine_E[num_min:num_max], fine_dos[num_min:num_max]

    def get_isovalue_scanning(self, data, accuracy=1e-5, scanner_type='interpolate', isovalue=1e-2):
        size = data.shape
        size_c = size[2]

        Nz_min = int(self.zmin / self.height * size_c)
        Nz_max = int(self.zmax / self.height * size_c)

        h_min = Nz_min / size_c * self.height
        h_max = Nz_max / size_c * self.height

        scanner = iso_scanner(scanner_type=scanner_type,
                              accuracy=accuracy,
                              isovalue=isovalue
                              )
        H = scanner.scanning(data,
                             Nz_max=Nz_max,
                             Nz_min=Nz_min,
                             h_max=h_max - self.surface_height,
                             h_min=h_min - self.surface_height
                             )

        return H

    def get_isoheight_scanning(self, data, isoheight=4.5):
        cell_height = isoheight + self.surface_height
        Nz = self.shape[2]

        index = int(cell_height / self.height * Nz)
        H = data[:, :, index]

        return H

    def get_STM_images(self, data, aperiod, bperiod, color_dict=None, size_dict=None, mode='auto', vmin=0.0, vmax=1.0, cell=None):
        if cell is None:
            fig, ax = contour_plot(data=data,
                                   aperiod=aperiod,
                                   bperiod=bperiod,
                                   cell=[self.cell[0], self.cell[1]],
                                   figure_name=self.figcounter,
                                   mode=mode,
                                   vmin=vmin,
                                   vmax=vmax,
                                   )
        else:
            fig, ax = contour_plot(data=data,
                                   aperiod=aperiod,
                                   bperiod=bperiod,
                                   cell=cell,
                                   figure_name=self.figcounter,
                                   mode=mode,
                                   vmin=vmin,
                                   vmax=vmax,
                                   )


        self.figcounter += 1

        X = []
        Y = []
        c = []
        s = []
        disp = int(aperiod / 2) * self.cell[0] + int(bperiod / 2) * self.cell[1]
        if color_dict is not None:
            if size_dict is not None:
                for i in range(0, len(self.atoms.get_chemical_symbols())):
                    element = self.atoms.get_chemical_symbols()[i]
                    if element in color_dict.keys():
                        color = color_dict[element]
                        size = size_dict[element]
                        pos = self.atoms.get_positions()[i] + disp
                        X += [pos[0]]
                        Y += [pos[1]]
                        c += [color]
                        s += [size]
                ax.scatter(X, Y, c=c, s=s)
            else:
                for i in range(0, len(self.atoms.get_chemical_symbols())):
                    element = self.atoms.get_chemical_symbols()[i]
                    if element in color_dict.keys():
                        color = color_dict[element]
                        pos = self.atoms.get_positions()[i] + disp
                        X += [pos[0]]
                        Y += [pos[1]]
                        c += [color]
                        s = 1.0
                ax.scatter(X, Y, c=c, s=s)
        else:
            if size_dict is not None:
                for i in range(0, len(self.atoms.get_chemical_symbols())):
                    element = self.atoms.get_chemical_symbols()[i]
                    if element in size_dict.keys():
                        size = size_dict[element]
                        pos = self.atoms.get_positions()[i] + disp
                        X += [pos[0]]
                        Y += [pos[1]]
                        s += [size]
                        c = 'r'
                ax.scatter(X, Y, c=c, s=s)
            else:
                pass

        return fig, ax
