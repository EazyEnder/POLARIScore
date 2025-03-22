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

VELOCITY_CHANNELS = 128
#resolution of 1km/s * 0.1
VELOCITY_RESOLUTION = 1e3*0.1
LSR_VELOCITY = 0
V = LSR_VELOCITY+(np.array(range(VELOCITY_CHANNELS))-VELOCITY_CHANNELS/2)*VELOCITY_RESOLUTION
DENSITY_THRESHOLD = 300

def compute_COSpectrum(simulation,position,direction,last_step,with_turb=True):
    temperature = simulation.data_temp[position[0],position[1],position[2]]
    def _get_velocity(pos):
        if not(all(0 <= pos[i] < simulation.nres for i in range(len(pos)))):
            return _get_velocity(position)[0], False
        v = np.array([simulation.data_vel[0][pos[0],pos[1],pos[2]],simulation.data_vel[1][pos[0],pos[1],pos[2]],simulation.data_vel[2][pos[0],pos[1],pos[2]]])
        return np.dot(v,direction)*1e3, True

    density = simulation.data[position[0],position[1],position[2]]

    if not("intensity_spectrum" in last_step):
        intensity_spectrum = np.zeros(V.shape)
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
    velocity, _ = _get_velocity(position)

    low_density_col = 1e4*simulation.cell_size.value*density*CO_ABUNDANCE/(1/3+2*temperature/5.5)
    tau0 = LIGHT_SPEED**3/(8*np.pi*CO_J10_FREQUENCY**3)*CO_J10_A * 3 * (1-np.exp(-CO_J10_TEMP/temperature)) * low_density_col
    tau = tau0 * GAUSSIAN(V-velocity,sigma)

    tau_exp = np.exp(-tau)
    intensity_spectrum = intensity_spectrum*tau_exp+BLACKBODY_EMISSION(nu=CO_J10_FREQUENCY,T=temperature)*(1-tau_exp) 

    result = {
        "intensity_spectrum": intensity_spectrum,
    }
    return result

def convertToKelvin(intensity_map):
    return intensity_map * LIGHT_SPEED**2 / (2.*BOLTZMANN_CONSTANT*CO_J10_FREQUENCY**2)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plotMap(intensity_map, slice=int(VELOCITY_CHANNELS/2), mean_mod=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if mean_mod:
        I = np.sum(intensity_map,axis=2)
        im = ax.imshow(I/VELOCITY_CHANNELS,extent=[simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]], cmap="jet", norm=LogNorm())
    else:
        im = ax.imshow(intensity_map[:,:,slice],extent=[simulation.axis[0][0], simulation.axis[0][1], simulation.axis[1][0],simulation.axis[1][1]], cmap="viridis")
    plt.colorbar(im, label="Intensity (K)")
    ax.set_xlabel(r"$x_1$ [pc]")
    ax.set_ylabel(r"$x_2$ [pc]")
    ax.legend()

    return fig, ax

def plotSpectrum(intensity_map, pos=(0,0), ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    intensity = intensity_map[pos[0],pos[1],:]
    ax.plot(V/1e3,intensity)

    return fig, ax

def saveSpectrum(intensity_map, name=None, replace=False):
    if name is None:
        name = "spectrum_"+str(uuid.uuid4())
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

def plot3D(data,threshold=15):
    data = data[::4, ::4, ::4]
    mask = data > threshold

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(data[mask])
    ax.voxels(mask, edgecolors="k")

    return fig, ax

if __name__ == "__main__":
    from objects.Simulation_DC import Simulation_DC
    simulation = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948)

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

    intensity_map = LoadSpectrum("spectrum_ff0f18eb-a9a1-47c1-8e84-a393ce8de3ef")
    plotMap(intensity_map, slice=int(VELOCITY_CHANNELS/2),mean_mod=True)
    #plotSpectrum(intensity_map)
    

    plt.show()


