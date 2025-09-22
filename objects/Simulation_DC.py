import os
import sys
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
from utils import *
from config import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import inspect
from batch_utils import compute_img_score
from astropy.io import fits
from astropy import units as u
import numpy as np
from objects.SpectrumMap import getSimulationSpectra
from objects.Dataset import Dataset
from typing import Dict,List,Tuple,Callable,Union
from matplotlib.widgets import Slider


class Simulation_DC():
    """
    DataCube Simulation is a sim where all the cells have the same size. 
    Easier to manipulate than AMR simulation, i.e the sim tree.
    """
    def __init__(self, name:str, global_size:float, init:bool=True):
        """
        DataCube Simulation is a sim where all the cells have the same size. 
        Easier to manipulate than AMR simulation, i.e the sim tree.

        Args:
            name (str): folder name where the simulation is stored
            global_size (float): size of the not cropped simulation in parsec
            init (bool, default:True): open files and load data, else need to call init() after. (For example after modifying the self.folder) 
        """
        self.name:str = name
        """Simulation name, name of the folder where the sim is in"""
        self.global_size:float = global_size
        """Real spatial size of the global simulation in parsec"""
        self.folder:str = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name+"/")
        """Path to the folder where the simulation is stored"""
        self.file:str = os.path.join(self.folder,SIM_DATA_NAME)
        """Path to the simulation data"""
        self.data:np.ndarray = None
        """Raw simulation density data"""
        self.data_temp:np.ndarray = None
        """Raw simulation temperature data"""
        self.data_vel:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        """Raw simulation velocity data (tuple of 3 datacube for xvel, yvel, zvel)"""

        self.header:Dict = None
        """Dict of sim settings"""
        self.nres:int = None
        """Resolution of the simulation (pixels*pixels), i.e shape of the matrix"""
        self.relative_size:float =None
        """Relative size of the simulation to the global simulation"""
        self.center:Tuple[float,float,float] = None
        """Center of the simulation to the global simulation"""
        self.cell_size:float = None
        """Simulation cell size in cm"""
        self.size:float = None
        """Real spatial size of the simulation in parsec"""
        self.axis:Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]] = None
        """Simulation faces surface in parsec"""

        """Cache for computed densities, ndarray are 2D tensors"""
        self.column_density:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        self.column_density_method:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        self.volumic_density:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]
        self.volumic_density_method:Tuple[np.ndarray,np.ndarray,np.ndarray] = [None,None,None]

        if init:
            self.init()

    def loadTemperature(self)->bool:
        """
        Load Temperature data from files

        Returns:
            isLoaded:bool
        """
        path = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_temp.fits")
        if not(os.path.exists(path)):
            LOGGER.warn(f"Temperature not loaded in {self.name}, file not found")
            return False
        simfile = fits.open(path)
        self.data_temp = simfile[0].data
        simfile.close()
        if self.data_temp is None:
            LOGGER.warn(f"Temperature not loaded in {self.name}, file empty")
            return False
        return True
    
    def loadVelocity(self)->bool:
        """
        Load velocity data from files

        Returns:
            isLoaded:bool
        """
        path_x = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_velx.fits")
        path_y = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_vely.fits")
        path_z = os.path.join(self.folder,SIM_DATA_NAME.split(".fits")[0]+"_velz.fits")

        if not(os.path.exists(path_x)):
            LOGGER.warn(f"Velocity not loaded in {self.name}, file for x component not found")
            return False
        if not(os.path.exists(path_y)):
            LOGGER.warn(f"Velocity not loaded in {self.name}, file for y component not found")
            return False
        if not(os.path.exists(path_z)):
            LOGGER.warn(f"Velocity not loaded in {self.name}, file for z component not found")
            return False
        
        simfile = fits.open(path_x)
        self.data_vel[0] = simfile[0].data
        simfile.close()
        simfile = fits.open(path_y)
        self.data_vel[1] = simfile[0].data
        simfile.close()
        simfile = fits.open(path_z)
        self.data_vel[2] = simfile[0].data
        simfile.close()

        return True

    def init(self, loadTemp:bool=False, loadVel:bool=False):
        """
        Load files and data in self variables

        Args:
            loadTemp (bool): try to load temperature ?
            loadVel (bool): try to load velocity ?
        """

        LOGGER.log(f"Loading simulation {self.name}")

        simfile = fits.open(self.file)
        self.data = simfile[0].data
        simfile.close()

        self.column_density = [None,None,None]
        self.column_density_method = [None,None,None]
        self.volumic_density = [None,None,None]
        self.volumic_density_method = [None,None,None]
        
        if loadTemp:
            LOGGER.log(f"Loading temperature of simulation {self.name}")
            self.loadTemperature()
        if loadVel:
            LOGGER.log(f"Loading velocity of simulation {self.name}")
            self.loadVelocity()

        if os.path.exists(os.path.join(self.folder,"processing_config.json")):
            with open(os.path.join(self.folder,"processing_config.json"), "r") as file:
                self.header = json.load(file)
            self.nres = self.header["run_parameters"]["nres"] if "nres" in self.header["run_parameters"] else self.header["run_parameters"]["nxyz"]
            self.relative_size = self.header["run_parameters"]["size"]
            self.center = np.array([self.header["run_parameters"]["xcenter"],self.header["run_parameters"]["ycenter"],self.header["run_parameters"]["zcenter"]])
            self.cell_size = (self.global_size*self.relative_size/self.nres) * u.parsec
            self.cell_size = self.cell_size.to(u.cm).value
            self.size = self.global_size*self.relative_size
            self.axis = ([self.center[0]*self.global_size-self.size/2,self.center[0]*self.global_size+self.size/2],[self.center[1]*self.global_size-self.size/2,self.center[1]*self.global_size+self.size/2],[self.center[2]*self.global_size-self.size/2,self.center[2]*self.global_size+self.size/2])    
        LOGGER.log(f"Loading finished for simulation {self.name}")

    def from_index_to_scale(self,index:int)->float:
        """Return the size in cm"""
        return index*self.cell_size.value

    def _compute_c_density(self, method:Callable=compute_column_density, axis:int=0, force:bool=False)->np.ndarray:
        """
        Compute column density of an axis if not already computed or force param is set to true.
        Args:
            method: Method used to compute the column density.
            axis (int): Axis
            force (bool): If true, then even if the column density was already computed on this face, this will be computed again.
        Returns:
            2D matrix (ndarray) 
        """
        if self.column_density_method[axis] is None or self.column_density_method[axis] != method.__name__ or self.column_density[axis] is None or force:
            LOGGER.log(f"Computing {method.__name__} for face {axis}, for {self.name}")
            self.column_density_method[axis] = method.__name__
            self.column_density[axis] = method(self.data, self.cell_size, axis=axis)
        return self.column_density[axis]
    
    def _compute_v_density(self, method:Callable=compute_mass_weighted_density, axis:int=0, force:bool=False)->np.ndarray:
        """
        Compute volume density of an axis if not already computed or force param is set to true.
        Args:
            method: Method used to compute the volume density.
            axis (int): Axis
            force (bool): If true, then even if the volume density was already computed on this face, this will be computed again.
        Returns:
            2D matrix (ndarray) 
        """
        if self.volumic_density_method[axis] is None or self.volumic_density_method[axis] != method.__name__ or self.volumic_density[axis] is None or force:
            LOGGER.log(f"Computing {method.__name__} for face {axis}, for {self.name}")
            self.volumic_density_method[axis] = method.__name__
            self.volumic_density[axis] = method(self.data, axis=axis)
        return self.volumic_density[axis]

    def generate_batch(self,name:str=None,method:Callable=compute_mass_weighted_density,what_to_compute:Dict={"cospectra":False,"density":False,"divide_vdens":False},number:int=8,size:float=5.,force_size:int=0,random_rotate:bool=True,limit_area:Tuple=([27,40,26,39],[26.4,40,22.5,44.3],[26.4,39,21,44.5]),nearest_size_factor:float=0.75)->bool:
        """
        Generate a batch, i.e pairs of images (2D matrix) like [(col_dens_1, vol_dens_1),(col_dens_2, vol_dens_2)]
        using this simulation. This will take randoms positions images in simulation.

        Args:
            method(function): Method to compute the volumic density, like do we take the volume weighted mean ? Or the mass weighted mean ? Or even the max density along the l.o.s ?
            number(int, default: 8): How many pairs of images do we want.
            size(float, default: 5): Size in parsec for the areas, this will be rounded to the nearest power of 2 pixels.
            force_size(int, default: 0): Force the size to be n pixels (for example 128).
            random_rotate(bool, default: True): Randomly rotate 0°,90°,180°,270° for each region.
            limit_area(list): In which region of the simulation we'll pick the areas: ([for face1],[for face2],[for face3]) -> ([x_min,x_max,y_min,y_max],...) for each face.
            nearest_size_factor(float, default:0.75): If the new area picked is too close to an old area of a factor nearest_size_factor*area_size then we'll choose another area.

        Returns:
            flag: if dataset was correctly generated.
        """

        LOGGER.border("BATCH-GENERATING")

        LOGGER.log(f"Generating {number} images using simulation {self.name}.")



        column_density = [self._compute_c_density(axis=0),self._compute_c_density(axis=1),self._compute_c_density(axis=2)]
        volume_density = [self._compute_v_density(method, axis=0),self._compute_v_density(method, axis=1),self._compute_v_density(method, axis=2)]

        flag_cospectra = what_to_compute["cospectra"] if "cospectra" in what_to_compute else False
        if flag_cospectra:
            co_spectra = getSimulationSpectra(self)
        flag_number_density = what_to_compute["density"] if "density" in what_to_compute else False
        flag_divide_vdens = what_to_compute["divide_vdens"] if "divide_vdens" in what_to_compute else False

        order = ["cdens","vdens"]
        if flag_cospectra:
            order.append("cospectra")
        if flag_number_density:
            order.append("density")
        if flag_divide_vdens:
            order.append("vdensdiffuse")
            order.append("vdensdense")

        name = self.name if name is None else name

        ds = Dataset()
        ds.name = name
        ds.settings = {"order": order}

        scores = []
        img_generated = 0
        areas_explored = [[],[],[]]
        iteration = 0
        while img_generated < number and iteration < number*100 :
            iteration += 1
            print(f'{iteration}', end = "\r")
            if iteration >= number*100:
                LOGGER.warn("Failed to generated all the requested random batches, nbr of imgs generated:"+str(img_generated))
                break

            face = int(np.floor(np.random.random()*3))
            c_dens = column_density[face]
            v_dens = volume_density[face]
            if flag_cospectra:
                co_spec = co_spectra[face]  
            
            limits = limit_area[face]
            if limits is None:
                limits = [0,self.global_size,0,self.global_size]
            center = np.array([limits[0]+(limits[1]-limits[0])*np.random.random(),limits[2]+(limits[3]-limits[2])*np.random.random()])

            s = int(2**round(np.log2(np.floor(convert_pc_to_index(size, self.nres, self.size)/2)*2)))
            if force_size > 0:
                s = int(force_size)

            size = self.from_index_to_scale(s)/PC_TO_CM
            #Verify if the region is already covered by a previous generated image
            flag = False
            for point in areas_explored[face]:
                if np.linalg.norm(center-point) < nearest_size_factor * size:
                    flag = True
                    break
            if flag:
                continue
            c_x, c_y = center
            c_x = convert_pc_to_index(c_x, self.nres,self.size,start=self.axis[0][0])
            c_y = convert_pc_to_index(c_y, self.nres,self.size,start=self.axis[0][0])

            start_x = c_x - s // 2
            start_y = c_y - s // 2
            end_x = c_x + s // 2 + s%2
            end_y = c_y + s // 2 + s%2

            if(start_x < 0 or start_y < 0 or end_x >= self.nres or end_y >= self.nres):
                continue

            def _process_img(img, k):
                p_img = img
                p_img = p_img[start_x:end_x, start_y:end_y]

                #Verify if there is no low density region (outside cloud) inside the area
                #if(((cropped_vdens < 10).sum()) > s*s*0.01), TODO:
                #    continue

                # Randomly choose a rotation (0, 90, 180, or 270 degrees)
                if random_rotate:
                    p_img = np.rot90(p_img, k, axes=(0,1))
                return p_img

            k = np.random.choice([0, 1, 2, 3])
            b = [_process_img(c_dens,k),_process_img(v_dens,k)]
            if flag_cospectra:
                b.append(_process_img(co_spec,k))

            if flag_number_density:
                if face == 0:
                    densities = self.data[:, start_x:end_x, start_y:end_y]
                elif face == 1:
                    densities = self.data[start_x:end_x, :, start_y:end_y]
                elif face == 2:
                    densities = self.data[start_x:end_x, start_y:end_y, :]

                if densities.shape[0] == self.nres:
                    densities = np.moveaxis(densities, 0, -1)
                elif densities.shape[1] == self.nres:
                    densities = np.moveaxis(densities, 1, -1)

                if random_rotate:
                    densities = np.rot90(densities, k, axes=(0,1))

                b.append(densities)

            if flag_divide_vdens:
                if not(hasattr(self, 'vdens_diff')):
                    self.vdens_diff = [None,None,None]
                if not(hasattr(self, 'vdens_dense')):
                    self.vdens_dense = [None,None,None]

                if self.vdens_diff[face] is None or self.vdens_dense[face] is None:
                    diff, dense = compute_volume_weighted_density(self.data, axis=face, divide=True)
                    self.vdens_diff[face] = diff
                    self.vdens_dense[face] = dense

                b.append(_process_img(self.vdens_diff[face],k))
                b.append(_process_img(self.vdens_dense[face],k))

            score = compute_img_score(b[0],b[1])
            if(np.random.random() > RANDOM_BATCH_SCORE_fct(score[0])):
                continue

            ds.save_batch(b, img_generated)
            del b
            scores.append(score)
            areas_explored[face].append(center)
            img_generated += 1

        print("")
        
        #Random permutation
        '''
        random_idx = np.random.permutation(len(imgs))
        r_imgs = []
        for r_id in random_idx:
            r_imgs.append(imgs[r_id])
        imgs = r_imgs
        r_scores = []
        for r_id in random_idx:
            r_scores.append(scores[r_id])
        imgs = r_imgs'
        '''
        scores = [r[0] for r in scores]
        
        settings = {
            "SIM_name":self.name,
            "method": method.__name__,
            "order": order,
            "what_was_computed": what_to_compute,
            "img_number": img_generated,
            "img_size": size,
            "areas_explored":str(areas_explored),
            "scores": str(scores),
            "scores_fct": inspect.getsourcelines(RANDOM_BATCH_SCORE_fct)[0][0],
            "scores_offset": str(RANDOM_BATCH_SCORE_offset),
            "number_goal": number,
            "iteration": iteration,
            "random_rotate": random_rotate,
        }
        ds.settings = settings
        ds.save_settings()

        LOGGER.log(f"New dataset {ds.name} saved")
    
        return ds
    
    def plotSlice(self, axis:int=0, slice:int=256, N_arrows:int=20, show_velocity:bool=True, enable_slider:bool=True):
            
            if not(axis in [0,1,2]):
                LOGGER.warn(f"Slice plot: Axis {axis} is not valid -> take the default axis: 0")

            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)

            velocity = self.data_vel if show_velocity else [None,None,None]

            artists = {'im': None, 'qui': None}

            Nx, Ny = self.data.shape[1], self.data.shape[2]
            x = np.arange(Ny)
            y = np.arange(Nx)
            X, Y = np.meshgrid(x, y)

            def _plotData(slice=slice):

                global im, qui

                density = self.data[slice,:,:]
                if axis == 1:
                    density = self.data[:,slice,:]
                elif axis == 2:
                    density = self.data[:,:,slice]

                if not(velocity[0] is None):
                    Ux = velocity[0][slice,:,:]
                    Uy = velocity[1][slice,:,:]
                    if axis == 1:
                        Ux = velocity[0][:,slice,:]
                        Uy = velocity[2][:,slice,:]
                    elif axis == 2:
                        Ux = velocity[1][:,:,slice]
                        Uy = velocity[2][:,:,slice]

                    step_x = max(Ny // N_arrows, 1)
                    step_y = max(Nx // N_arrows, 1)

                    X_sub = X[::step_y, ::step_x]
                    Y_sub = Y[::step_y, ::step_x]
                    Ux_sub = Ux[::step_y, ::step_x]
                    Uy_sub = Uy[::step_y, ::step_x]

                if artists["im"] is None:
                    artists["im"] = ax.imshow(density, origin="lower", cmap="jet", extent=[0, Ny, 0, Nx], norm=LogNorm())
                else:
                    artists["im"].set_data(density)

                if not(velocity[0] is None):
                    if artists["qui"] is None:
                        artists["qui"] = ax.quiver(X_sub, Y_sub, Ux_sub, Uy_sub, color="white", scale=200)
                    else:
                        artists["qui"].set_UVC(Ux_sub, Uy_sub)

                ax.set_title(f"Slice {slice}")
                fig.canvas.draw_idle()

            _plotData(slice=slice)
            plt.colorbar(artists["im"], ax=ax, label="Density")

            if enable_slider:
                ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
                slider = Slider(ax_slider, 'Slice', 0, self.data.shape[0]-1, valinit=slice, valfmt='%0.0f')

                def update_slice(val):
                    slice_idx = int(slider.val)
                    _plotData(slice=slice_idx)

                slider.on_changed(update_slice)

    def plot(self,method:Callable=compute_column_density,axis:Union[List[int],int]=[0],plot_pdf:bool=False,color_bar:bool=True,derivate:int=0)->Tuple:
        """
        Plot simulations faces with probabiliy density function

        Args:
            method(function): Method to compute the data (2d tensor)
            axis(list or int): axis or axes
            plot_pdf(bool): if True plot the probability density function
            color_bar(bool): if True, plot the colorbar
            derivate(int): Derivate the data n times where n is derivate param.
        Returns:
            Tuple(fig, axes)
        """

        axis = axis if type(axis) is list else [axis]
        axis = np.array(axis)
        axis = axis[np.argsort(axis)]

        densities = []
        for ax in axis:
            d = method(self.data, self.cell_size, axis=ax)
            d = compute_derivative(d, order=derivate)
            d = np.abs(d)
            densities.append(d)  

        fig, axes = plt.subplots(2 if plot_pdf else 1, len(axis), figsize=(4*len(axis), 6 if plot_pdf else 3.5))
        if len(axis) <= 1:
            axes = [axes]
        if not(plot_pdf):
            axes = [axes]

        def _plot(column, data):
            cd = axes[0][column].imshow(data, extent=[self.axis[0][0], self.axis[0][1], self.axis[1][0],self.axis[1][1]], cmap="jet", norm=LogNorm())
            if plot_pdf:
                pdf = compute_pdf(data)
                axes[1][column].plot([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
                axes[1][column].scatter([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
                axes[1][column].set_xlabel("s")
                axes[1][column].set_ylabel("p")
                axes[1][column].set_title("PDF")
            return cd

        for i,_ in enumerate(axis):
            cd = _plot(0,densities[i])
            axes[0][i].set_title("Top-Down View (XY Projection)")
            axes[0][i].set_xlabel("X [pc]")
            axes[0][i].set_ylabel("Y [pc]")

        if color_bar:
            cbar = plt.colorbar(cd, ax=axes[0], orientation="vertical", fraction=0.02, pad=0.02)
            cbar.set_label("Column density ($cm^{-2}$)")

        return fig, axes

    def plot_correlation(self,method:Callable=compute_mass_weighted_density, axis:int=-1, contour_levels:int=0, force_compute:bool=False, lines:List[int]=[0,1,2])->Tuple:

        """
        Plot correlation between the column density and the volumic density

        Args:
            method(function): Method to compute volumic density
            axis(int): which face of the sim, if -1 all faces are taken
            contour_levels(int): If instead of using color map, a contour map is used (for value > 0, levels of the contour map = this var)
            force_compute(bool): if True, the column density and volume density will be computed even if cache is available.
        Returns:
            Tuple(fig, ax)
        """
        fig, ax = plt.subplots(1,1)
        if axis >= 0:
            column_density = np.log(self._compute_c_density(axis=axis,force=force_compute).flatten())/np.log(10)
            volume_density = np.log(self._compute_v_density(method=method, axis=axis,force=force_compute).flatten())/np.log(10)
        else:
            column_density = np.log(np.array([self._compute_c_density(axis=0,force=force_compute),self._compute_c_density(axis=1,force=force_compute),self._compute_c_density(axis=2,force=force_compute)]).flatten())/np.log(10)
            volume_density = np.log(np.array([self._compute_v_density(method=method, axis=0,force=force_compute),self._compute_v_density(method=method, axis=1,force=force_compute),self._compute_v_density(method=method, axis=2,force=force_compute)]).flatten())/np.log(10)

        if contour_levels > 1:
            hist, xedges, yedges = np.histogram2d(column_density, volume_density, bins=(256, 256))
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xcenters, ycenters)
            contour = ax.contour(X, Y, hist.T, levels=int(contour_levels), norm=LogNorm(), colors="black")
            ax.clabel(contour, fmt=lambda x: r"$10^{{{:.0f}}}$".format(np.log10(x)), inline=True, fontsize=8)
        else:
            _, _,_,hist = ax.hist2d(column_density, volume_density, bins=(256,256), norm=LogNorm())
            plt.colorbar(hist, ax=ax, label="counts")
        ax.set_xlabel(r"Column density ($log_{10}(cm^{-2})$)")
        ax.set_ylabel(r"Mass-weighted density ($log_{10}(cm^{-3})$)")

        ax = plt.gca()

        plot_lines(column_density, volume_density, ax, lines=lines)

        ax.grid(True)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return fig, ax

def mergeSimu(sim_array:List[Simulation_DC])->Simulation_DC:
    """
    Merge simulations into one.
    """
    assert all(sim.nres == sim_array[0].nres for sim in sim_array),  LOGGER.error("Resolution mismatch among simulations.")
    LOGGER.log(f"merge {len(sim_array)} simulations")
    datacube_size = sim_array[0].nres
    sim_len = int(len(sim_array)**(1/3))
    merged_simulation = np.zeros((datacube_size*sim_len, datacube_size*sim_len, datacube_size*sim_len))

    centers = np.array([sim.center for sim in sim_array])
    maxs = np.max(centers+sim_array[0].relative_size/2, axis=0)
    mins = np.min(centers-sim_array[0].relative_size/2, axis=0)

    for i,sim in enumerate(sim_array):
        printProgressBar(i, len(sim_array), prefix="Merging:")
        sim_data = sim.data.transpose()
        x_center, y_center, z_center = (centers[i] - mins)/(maxs-mins)
        x_offset = int((x_center) * datacube_size*sim_len -datacube_size/2)
        y_offset = int((y_center) * datacube_size*sim_len -datacube_size/2)
        z_offset = int((z_center) * datacube_size*sim_len -datacube_size/2)
        
        if 0 <= x_offset < datacube_size*sim_len and 0 <= y_offset < datacube_size*sim_len and 0 <= z_offset < datacube_size*sim_len:
            merged_simulation[x_offset:x_offset + datacube_size, 
                               y_offset:y_offset + datacube_size, 
                               z_offset:z_offset + datacube_size] = sim_data
        else:
            LOGGER.error(f"Simulation center ({x_center}, {y_center}, {z_center}) out of bounds for merged datacube.")
            return
    LOGGER.log("Simulations merged")

    host = sim_array[0]
    host.data = merged_simulation.transpose()
    host.nres = datacube_size*sim_len
    host.relative_size = host.relative_size*sim_len
    host.center = np.array([0.5,0.5,0.5])
    host.cell_size = (host.global_size*host.relative_size/host.nres) * u.parsec
    host.cell_size = host.cell_size.to(u.cm)
    host.size = host.global_size*host.relative_size
    host.axis = ([host.center[0]*host.global_size-host.size/2,host.center[0]*host.global_size+host.size/2],[host.center[1]*host.global_size-host.size/2,host.center[1]*host.global_size+host.size/2],[host.center[2]*host.global_size-host.size/2,host.center[2]*host.global_size+host.size/2])  
    
    return host

import glob
def openSimulation(name_root:str, global_size:float, use_cache:bool=True)->Simulation_DC:
    """
    Open a datacube simulation
    Args:
        name_root(str): name of the simulation folder
        global_size(float): physical size of the global simulation (not the datacube) like the datacube can be 5pc long but the simulation was runned with a grid 66pc long.
        use_cache(bool): Use cache
    Returns:
        Simulation
    """
    files =glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/sims/"+name_root+"*"))
    LOGGER.log(f"Opening {len(files)} simulations")
    names = [f.split("/")[-1] for f in files]
    sims = []
    if use_cache and os.path.exists(CACHES_FOLDER+"np_memory"):
        LOGGER.log("Merge using cached data")
        sim = Simulation_DC(names[0], global_size, init=True)
        sim_len = int(len(names)**(1/3))
        sim.data = np.memmap(CACHES_FOLDER+"np_memory", dtype='float32', mode='r', shape=(sim.data.shape[0]*sim_len,sim.data.shape[1]*sim_len,sim.data.shape[2]*sim_len))
        sim.nres = sim.nres*sim_len
        sim.relative_size = sim.relative_size*sim_len
        sim.center = np.array([0.5,0.5,0.5])
        sim.cell_size = (sim.global_size*sim.relative_size/sim.nres) * u.parsec
        sim.cell_size = sim.cell_size.to(u.cm)
        sim.size = sim.global_size*sim.relative_size
        sim.axis = ([sim.center[0]*sim.global_size-sim.size/2,sim.center[0]*sim.global_size+sim.size/2],[sim.center[1]*sim.global_size-sim.size/2,sim.center[1]*sim.global_size+sim.size/2],[sim.center[2]*sim.global_size-sim.size/2,sim.center[2]*sim.global_size+sim.size/2])  
        return sim

    for n in names:
        sims.append(Simulation_DC(n, global_size, init=True))
    sim = mergeSimu(sims)
    del sims
    fp = np.memmap(CACHES_FOLDER+"np_memory", dtype='float32', mode='w+', shape=sim.data.shape)
    fp[:] = sim.data[:]
    sim.data = fp
    return sim

if __name__ == "__main__":
    sim = Simulation_DC(name="orionMHD_lowB_0.39_512", global_size=66.0948, init=False)
    sim.init(loadTemp=True, loadVel=True)
    #sim = openSimulation("orionMHD_lowB_multi", global_size=66.0948)
    #sim.plot(derivate=2, axis=0)
    #plt.figure()
    #sim.plot_correlation(method=compute_mass_weighted_density, contour_levels=3)
    
    #sim.generate_batch(name="orionMHD_lowB_0.39_512_13CO_max",method=compute_max_density,what_to_compute = {"cospectra":True}, number = 1000, force_size=128, nearest_size_factor=0.75)
    #from Dataset import getDataset
    #ds = getDataset("batch_highres_twochannels")
    #pair = ds.get(1)
    #indexes = ds.get_element_index(["vdens","vdensdiffuse","vdensdense","cdens"])
     
    #norm = LogNorm(vmin=np.min(pair[indexes[0]]),vmax=np.max(pair[indexes[0]]))
    #plt.figure()
    #plt.imshow(pair[indexes[0]], norm= LogNorm())
    #plt.figure()
    #plt.imshow(pair[indexes[1]], norm= LogNorm())
    #plt.figure()
    #plt.imshow(pair[indexes[2]], norm= LogNorm())

    plt.show()