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
VELOCITY_RESOLUTION = 1e3*0.5
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

def plotMap(intensity_map, simulation, slice=int(VELOCITY_CHANNELS/2), mean_mod=False, ax=None, norm=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if mean_mod:
        I = np.sum(intensity_map,axis=2)
        im = ax.imshow(I/VELOCITY_CHANNELS,extent=[simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]], cmap="jet", norm=LogNorm() if norm is None else norm)
    else:
        im = ax.imshow(intensity_map[:,:,slice],extent=[simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]], cmap="viridis")
    plt.colorbar(im, label="Intensity (K)")
    ax.set_xlabel(r"$x_1$ [pc]")
    ax.set_ylabel(r"$x_2$ [pc]")
    ax.legend()

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

def script_create_spectrum(simulation, region=[254,256,254,256], name=None):
    results = ray_mapping(simulation, compute_COSpectrum, axis=1, region=[0,-1,0,-1])
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

if __name__ == "__main__":
    from objects.Simulation_DC import Simulation_DC
    #simulation = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948, init=False)
    #sim = openSimulation("orionMHD_lowB_multi_", global_size=66.0948, use_cache=True)
    #simulation.init(loadTemp=True,loadVel=True)

    #intensity_map = script_create_spectrum(simulation, name="32_withCMB_Y")
    #plotSpectrum(intensity_map, pos=(255,255))
    #plotMap(intensity_map, slice=int(VELOCITY_CHANNELS/2),mean_mod=False, norm=LogNorm(vmin=1e-3,vmax=20))

    intensity_map = LoadSpectrum("spectrum_orionMHD_lowB_0.39_512_1")

    """
    results = ray_mapping(simulation, compute_COSpectrum, axis=0, region=[0,50,0,50])
    intensity_map = []
    for i in range(len(results)):
        intensity_map.append([])
        for j in range(len(results[i])):
            intensity_map[i].append(results[i][j]["intensity_spectrum"])
    intensity_map = np.array(intensity_map)
    intensity_map = convertToKelvin(intensity_map)
    saveSpectrum(intensity_map)"
    """

    #intensity_map = LoadSpectrum("32_withCMB")
    plotSpectrum(intensity_map, pos=(255,255))
    plt.figure()
    plt.imshow(getIntegratedIntensity(intensity_map))
    #plotMap(intensity_map, slice=int(VELOCITY_CHANNELS/2),mean_mod=False, norm=LogNorm(vmin=1e-3,vmax=20))
    #intensity_map = LoadSpectrum("spectrum_e271a125-d774-420a-81aa-5bcec00c6053")
    #plotSpectrum(intensity_map, pos=(255,255), v_channels=128, v_res=1e3*0.25)
    #plotMap(intensity_map, slice=int(VELOCITY_CHANNELS/2),mean_mod=False, norm=LogNorm(vmin=1e-3,vmax=20))

    plt.show()


