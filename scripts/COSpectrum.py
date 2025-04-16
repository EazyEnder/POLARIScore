import sys
import os
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

from objects.Raycaster import ray_mapping
from physics_utils import *
import numpy as np
from config import *
import uuid

#Output settings
VELOCITY_CHANNELS = 256
#resolution of 1km/s * x
VELOCITY_RESOLUTION = 1e3*0.1
LSR_VELOCITY = 0
V = LSR_VELOCITY+(np.array(range(VELOCITY_CHANNELS))-VELOCITY_CHANNELS/2)*VELOCITY_RESOLUTION
DENSITY_THRESHOLD = 300

#Line settings example for 12CO J=U-L
L = 0
U = 1
LINE_SETTINGS = {
    "l":L,
    "u":U,
    "abundance":CO_ABUNDANCE,
    "temp_low":ROT_ENERGY(L,CO_ROT_CST),
    "temperature":ROT_ENERGY(U,CO_ROT_CST)-ROT_ENERGY(L,CO_ROT_CST),
    "frequency":CO_FREQUENCY[U-1],
    "estein_emission":CO_A[U-1]
}
if __name__ == "__main__":
    LOGGER.log(f"Line settings used: {LINE_SETTINGS}")

def compute_COSpectrum(simulation,position,direction,last_step, with_turb=True, line_settings=LINE_SETTINGS):
    temperature = simulation.data_temp[position[0],position[1],position[2]]

    g_l = 2*L+1
    g_u = 2*U+1

    def _get_velocity(pos):
        if not(all(0 <= pos[i] < simulation.nres for i in range(len(pos)))):
            return _get_velocity(position)[0], False
        v = np.array([simulation.data_vel[0][pos[0],pos[1],pos[2]],simulation.data_vel[1][pos[0],pos[1],pos[2]],simulation.data_vel[2][pos[0],pos[1],pos[2]]])
        return np.dot(v,direction)*1e3, True

    density = simulation.data[position[0],position[1],position[2]]
    velocity, _ = _get_velocity(position)

    if not("intensity_spectrum" in last_step):
        intensity_spectrum = BLACKBODY_EMISSION((V/LIGHT_SPEED+1)*line_settings["frequency"],CMB_TEMPERATURE)
    else:
        intensity_spectrum = last_step["intensity_spectrum"]

    if density < DENSITY_THRESHOLD:
        return {"intensity_spectrum": intensity_spectrum}


    sigma_doppler = 0.08*1e3*np.sqrt(temperature/20)
    sigma_turb = 0
    if with_turb:
        vm1,fm1 = _get_velocity((position-direction).astype(int))
        vp1,fp1 = _get_velocity((position+direction).astype(int))
        sigma_turb = 0.5*(vp1-vm1) if fm1 and fp1 else vp1-vm1
    sigma = np.sqrt(sigma_doppler**2 + sigma_turb**2)

    low_density_col = 1e4*simulation.cell_size.value*density*line_settings["abundance"]*g_l*np.exp(-line_settings["temp_low"]/(temperature))/(1/3+2*temperature/5.5)
    tau0 = LIGHT_SPEED**3/(8*np.pi*line_settings["frequency"]**3)*line_settings["estein_emission"] * g_u/g_l * (1-np.exp(-line_settings["temperature"]/temperature)) * low_density_col
    tau = tau0 * GAUSSIAN(V-velocity,sigma)

    tau_exp = np.exp(-tau)
    intensity_spectrum = intensity_spectrum*tau_exp+BLACKBODY_EMISSION(nu=line_settings["frequency"],T=temperature)*(1-tau_exp) 

    result = {
        "intensity_spectrum": intensity_spectrum,
    }
    return result

def convertToKelvin(intensity_map,frequency):
    return intensity_map * LIGHT_SPEED**2 / (2.*BOLTZMANN_CONSTANT*frequency**2)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider

def plotMap(intensity_map, simulation=None, slice=int(VELOCITY_CHANNELS/2), mean_mod=False, ax=None, norm=None, enable_slider=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if mean_mod:
        I = np.sum(intensity_map,axis=2)
        im = ax.imshow(I/VELOCITY_CHANNELS,extent=None if simulation is None else [simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]] , cmap="jet", norm=LogNorm() if norm is None else norm)
    else:
        im = ax.imshow(intensity_map[:,:,slice],extent=None if simulation is None else [simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]], cmap="viridis")
    plt.colorbar(im, label="Intensity (K)")
    ax.set_xlabel(r"$x_1$ [pc]")
    ax.set_ylabel(r"$x_2$ [pc]")
    ax.legend()

    if not(mean_mod) and enable_slider:
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, intensity_map.shape[2] - 1, valinit=slice, valfmt='%0.0f')

        def update_slice(val):
            slice_idx = int(slider.val)
            im.set_data(intensity_map[:,:,slice_idx])
            fig.canvas.draw_idle()

        slider.on_changed(update_slice)

    return fig, ax

def plotSpectrum(intensity_map, pos=(0,0), ax=None, v_channels=VELOCITY_CHANNELS, v_res=VELOCITY_RESOLUTION):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    V = LSR_VELOCITY+(np.array(range(v_channels))-v_channels/2)*v_res

    intensity = intensity_map[pos[0],pos[1],:]
    ax.plot(V/1e3,intensity)

    return fig, ax

def plot(intensity_map):
    fig, ax = plt.subplots()
    image = ax.imshow(getIntegratedIntensity(intensity_map))

    fig2, ax2 = plt.subplots()
    plotSpectrum(intensity_map, ax=ax2, pos=(255, 255))

    def onclick(event):
        if event.inaxes == ax:
            y = int(round(event.xdata))
            x = int(round(event.ydata))
            ax2.cla()
            #data, data_fit = fit_gaussians(intensity_map[x,y,:])
            #plot_fit(data,data_fit, ax=ax2)
            custom_fit(intensity_map[x,y,:], ax=ax2)
            #plotSpectrum(intensity_map, ax=ax2, pos=(x, y))
            ax2.set_title(f"Spectrum at ({x}, {y})")
            plt.show()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def saveSpectrum(intensity_map, name=None, replace=False):
    if name is None:
        name = "spectrum_"+str(uuid.uuid4())
    else:
        if not("spectrum" in name):
            name = "spectrum_"+name

    if not(os.path.exists(CACHES_FOLDER)):
        os.mkdir(CACHES_FOLDER)
    path = os.path.join(CACHES_FOLDER,name+".npy")
    if os.path.exists(path):
        if not(replace):
            LOGGER.error(f"Can't save spectrum {name} because there is already a cache and replace is set to False")
            return
        os.remove(path)
    LOGGER.log(f"Spectrum {name} saved")
    np.save(path,intensity_map)

def LoadSpectrum(name):
    path = os.path.join(CACHES_FOLDER,name.split(".npy")[0]+".npy")
    if not(os.path.exists(path)):
        return 
    return np.load(path)

def getSimulationSpectra(simulation):

    name = simulation.name
    spectra = [LoadSpectrum("spectrum_"+name+"_"+str(int(i+1))) for i in range(3)]
    for i,s in enumerate(spectra):
        if s is None:
            LOGGER.log(f"Spectrum for face {i} doesn't exist, generating it: ")
            results = ray_mapping(simulation, compute_COSpectrum, axis=i, region=[0,-1,0,-1])
            intensity_map = []
            for k in range(len(results)):
                intensity_map.append([])
                for j in range(len(results[i])):
                    intensity_map[k].append(results[k][j]["intensity_spectrum"])
            intensity_map = intensity_map-BLACKBODY_EMISSION(((V)/LIGHT_SPEED+1)*LINE_SETTINGS["frequency"],CMB_TEMPERATURE)
            intensity_map = np.array(intensity_map)
            intensity_map = convertToKelvin(intensity_map,LINE_SETTINGS["frequency"])
            saveSpectrum(intensity_map, name=name+"_"+str(i+1))
            spectra[i] = intensity_map
    return spectra

def getIntegratedIntensity(intensity_map):
    intensity_map = np.array(intensity_map)
    return np.sum(intensity_map, axis=2)

def plot3D(data,threshold=15):
    data = data[::4, ::4, ::4]
    mask = data > threshold

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(data[mask])
    ax.voxels(mask, edgecolors="k")

    return fig, ax

def script_create_spectrum(simulation, region=[254,256,254,256], name=None, axis=0):
    results = ray_mapping(simulation, compute_COSpectrum, axis=axis, region=[0,-1,0,-1])
    intensity_map = []
    for i in range(len(results)):
        intensity_map.append([])
        for j in range(len(results[i])):
            intensity_map[i].append(results[i][j]["intensity_spectrum"])
    intensity_map = intensity_map-BLACKBODY_EMISSION(((V)/LIGHT_SPEED+1)*LINE_SETTINGS["frequency"],CMB_TEMPERATURE)
    intensity_map = np.array(intensity_map)
    intensity_map = convertToKelvin(intensity_map,LINE_SETTINGS["frequency"])
    saveSpectrum(intensity_map, name=name)
    return intensity_map

import pickle
import gausspy.gp as gp
def fit_gaussians(spectrum):
    channels = np.arange(len(spectrum))
    errors = np.ones(channels.shape)*0.1

    data = {}
    data['data_list'] = data.get('data_list', []) + [spectrum]
    data['x_values'] = data.get('x_values', []) + [channels]
    data['errors'] = data.get('errors', []) + [errors]
    FILENAME = CACHES_FOLDER+"spectrum.pickle"
    pickle.dump(data, open(FILENAME, 'wb'))
    g = gp.GaussianDecomposer()
    g.set('phase', 'one')
    g.set('SNR_thresh', [0.1])
    g.set('alpha1',0.0001)
    data_decomp = g.batch_decomposition(FILENAME)
    return data,data_decomp

def plot_fit(data,data_decomp,ax=None):
    def _gaussian(amp, fwhm, mean):
        return lambda x: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)

    def _unravel(list):
        return np.array([i for array in list for i in array])
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    spectrum = _unravel(data['data_list'])
    chan = _unravel(data['x_values'])
    errors = _unravel(data['errors'])

    means_fit = _unravel(data_decomp['means_fit'])
    amps_fit = _unravel(data_decomp['amplitudes_fit'])
    fwhms_fit = _unravel(data_decomp['fwhms_fit'])

    model = np.zeros(len(chan))

    for j in range(len(means_fit)):
        component = _gaussian(amps_fit[j], fwhms_fit[j], means_fit[j])(chan)
        model += component
        ax.plot(chan, component, color='red', lw=1.5)

    ax.plot(chan, spectrum, label='Data', color='black', linewidth=1.5)
    ax.plot(chan, model, label = r'$\log\alpha=1.$', color='purple', linewidth=2.)
    ax.plot(chan, errors, label = 'Errors', color='green', linestyle='dashed', linewidth=2.)

    ax.set_xlabel('Channels')
    ax.set_ylabel('Amplitude')

    ax.set_xlim(0,len(chan))
    ax.set_ylim(np.min(spectrum),np.max(spectrum))
    ax.legend(loc=2)

    return fig, ax

def gaussian_sum(x, params, N):
    y = np.zeros_like(x)
    for i in range(N):
        A, mu, sigma = params[3*i], params[3*i+1], params[3*i+2]
        y += np.abs(A) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    return y

from scipy.optimize import minimize
def custom_fit(Y, max_N=5, ax=None):
    X = np.arange(len(Y), dtype=float)

    def chi_squared(params, x, y, N):
        y_model = gaussian_sum(x, params, N)
        return np.sum((y - y_model)**2/(y_model+1e-8))
    
    best_result = None
    results = []

    best_bic = np.inf
    for N in range(1, max_N+1):
        guess = []
        for i in range(N):
            guess.extend([np.random.uniform(min(Y), max(Y)), X[int(len(X)/2)], np.random.uniform(1,10)])
        
        res = minimize(chi_squared, guess, args=(X, Y, N), method='L-BFGS-B')
        k = len(res.x)
        chi2 = chi_squared(res.x, X, Y, N)
        n = len(X)
        bic = 2.*k + chi2
        results.append((N, res, bic))
        print(N,bic)
        if bic < best_bic:
            best_bic = bic
            best_result = (N, res)

    N_best, res = best_result
    y_fit = gaussian_sum(X, res.x, N_best)

    if not(ax is None):
        ax.plot(X, Y, label='Data')
        ax.plot(X, y_fit, 'r-', label=f'Fit (N={N_best})')
        ax.legend()
        ax.set_title(f'Best BIC model: N = {N_best}')

    return (N_best, res.x)




if __name__ == "__main__":
    from objects.Simulation_DC import Simulation_DC
    simulation = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948, init=False)
    #sim = openSimulation("orionMHD_lowB_multi_", global_size=66.0948, use_cache=True)
    simulation.init(loadTemp=True,loadVel=True)

    intensity_map = LoadSpectrum("spectrum_orionMHD_lowB_0.39_512_1")

    #pos = (255,255)
    #plotSpectrum(intensity_map, pos=pos)
    from utils import movingMin, movingAverage
    #Y = intensity_map[pos[0],pos[1],:]
    #custom_fit(Y, max_N=10, plot=True)

    plot(intensity_map)

    #plotMap(intensity_map, simulation=simulation, slice=int(VELOCITY_CHANNELS/2),mean_mod=False, norm=LogNorm(vmin=1e-3,vmax=20))
    #simulation.plotSlice(show_velocity=False, axis=2)
    #plot(intensity_map)
    #simulation.plot()

    #plotSpectrum(intensity_map, pos=(255,255))

    plt.show()


