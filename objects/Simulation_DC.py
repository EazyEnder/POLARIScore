from utils import *
from config import *
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import inspect
from training_batch import compute_img_score
from astropy.io import fits
from astropy import units as u
import numpy as np

class Simulation_DC():
    """
    DataCube Simulation is a sim where all the cells have the same size. 
    Easier to manipulate than AMR simulation, i.e the sim tree.
    """
    def __init__(self, name, global_size, init=True):
        """
        DataCube Simulation is a sim where all the cells have the same size. 
        Easier to manipulate than AMR simulation, i.e the sim tree.

        Args:
            name (str): folder name where the simulation is stored
            global_size (float): size of the not cropped simulation in parsec
            init (bool, default:True): open files and load data, else need to call init() after. (For example after modifying the self.folder) 
        """
        self.name = name
        """Simulatio name, name of the folder where the sim is in"""
        self.global_size = global_size
        """Real spatial size of the global simulation in parsec"""
        self.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../sims/"+name+"/")
        """Path to the folder where the simulation is stored"""
        self.file = os.path.join(self.folder,SIM_DATA_NAME)
        """Path to the simulation data"""
        self.data = None
        """Raw simulation density data"""
        self.data_temp = None
        """Raw simulation temperature data"""
        self.data_vel = [None,None,None]
        """Raw simulation velocity data (tuple of 3 datacube for xvel, yvel, zvel)"""

        self.header = None
        """Dict of sim settings"""
        self.nres = None
        """Resolution of the simulation (pixels*pixels), i.e shape of the matrix"""
        self.relative_size =None
        """Relative size of the simulation to the global simulation"""
        self.center = None
        """Center of the simulation to the global simulation"""
        self.cell_size = None
        """Simulation cell size in cm"""
        self.size = None
        """Real spatial size of the simulation in parsec"""
        self.axis = None
        """Simulation area in parsec"""

        self.column_density = [None,None,None]
        self.column_density_method = [None,None,None]
        self.volumic_density = [None,None,None]
        self.volumic_density_method = [None,None,None]

        if init:
            self.init()

    def loadTemperature(self):
        """
        Load Temperature data from files
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
    
    def loadVelocity(self):
        """
        Load velocity data from files
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

    def init(self, loadTemp=True, loadVel=True):
        """
        Load files and data in self variables
        """
        simfile = fits.open(self.file)
        self.data = simfile[0].data
        simfile.close()
        
        if loadTemp:
            self.loadTemperature()
        if loadVel:
            self.loadVelocity()

        with open(os.path.join(self.folder,"processing_config.json"), "r") as file:
            self.header = json.load(file)
        self.nres = self.header["run_parameters"]["nres"] if "nres" in self.header["run_parameters"] else self.header["run_parameters"]["nxyz"]
        self.relative_size = self.header["run_parameters"]["size"]
        self.center = np.array([self.header["run_parameters"]["xcenter"],self.header["run_parameters"]["ycenter"],self.header["run_parameters"]["zcenter"]])
        self.cell_size = (self.global_size*self.relative_size/self.nres) * u.parsec
        self.cell_size = self.cell_size.to(u.cm)
        self.size = self.global_size*self.relative_size
        self.axis = ([self.center[0]*self.global_size-self.size/2,self.center[0]*self.global_size+self.size/2],[self.center[1]*self.global_size-self.size/2,self.center[1]*self.global_size+self.size/2],[self.center[2]*self.global_size-self.size/2,self.center[2]*self.global_size+self.size/2])    

    def from_index_to_scale(self,index):
        """Return the size in cm"""
        return index*self.cell_size.value

    def _compute_c_density(self, method=compute_column_density, axis=0, force=False):
        if self.column_density_method[axis] is None or self.column_density_method[axis] != method.__name__ or self.column_density[axis] is None or force:
            self.column_density_method[axis] = method.__name__
            self.column_density[axis] = method(self.data, self.cell_size, axis=axis)
        return self.column_density[axis]
    
    def _compute_v_density(self, method=compute_mass_weighted_density, axis=0, force=False):
        if self.volumic_density_method[axis] is None or self.volumic_density_method[axis] != method.__name__ or self.volumic_density[axis] is None or force:
            self.volumic_density_method[axis] = method.__name__
            self.volumic_density[axis] = method(self.data, axis=axis)
        return self.volumic_density[axis]

    def generate_batch(self,method=compute_mass_weighted_density,number=8,size=5,force_size=0,random_rotate=True,limit_area=([27,40,26,39],[26.4,40,22.5,44.3],[26.4,39,21,44.5]),nearest_size_factor=0.75):
        """
        Generate a batch, i.e pairs of images (2D matrix) like [(col_dens_1, vol_dens_1),(col_dens_2, vol_dens_2)]
        using this simulation. This will take randoms positions images in simulation.

        Args:
            method(function): Method to compute the volumic density, like do we take the volume weighted mean ? Or the mass weighted mean ? Or even the max density along the l.o.s ?
            number(float, default: 8): How many pairs of images do we want.
            size(float, default: 5): Size in parsec for the areas, this will be rounded to the nearest power of 2 pixels.
            force_size(float, default: 0): Force the size to be n pixels (for example 128).
            random_rotate(bool, default: True): Randomly rotate 0°,90°,180°,270° for each region.
            limit_area(list): In which region of the simulation we'll pick the areas: ([for face1],[for face2],[for face3]) -> ([x_min,x_max,y_min,y_max],...) for each face.
            nearest_size_factor(float, default:0.75): If the new area picked is too close to an old area of a factor nearest_size_factor*area_size then we'll choose another area.

        Returns:
            tuple: (batch,settings) where settings is a log dict.
        """

        LOGGER.log(f"Generate {number} images using simulation {self.name}.")

        column_density_xy = self._compute_c_density(axis=0)
        column_density_xz = self._compute_c_density(axis=1)
        column_density_yz = self._compute_c_density(axis=2)
        volume_density_xy = self._compute_v_density(method, axis=0)
        volume_density_xz = self._compute_v_density(method, axis=0)
        volume_density_yz = self._compute_v_density(method, axis=0)

        imgs = []
        scores = []
        img_generated = 0
        areas_explored = [[],[],[]]
        iteration = 0
        while img_generated < number and iteration < number*100 :
            iteration += 1
            if iteration >= number*100:
                LOGGER.warn("Failed to generated all the requested random batches, nbr of imgs generated:"+str(len(imgs)))
                break

            random = np.random.random()
            c_dens = column_density_xy
            v_dens = volume_density_xy
            face = 0
            if random < 1/3:
                c_dens = column_density_xz
                v_dens = volume_density_xz
                face = 1
            elif random < 2/3:
                c_dens = column_density_yz
                v_dens = volume_density_yz
                face = 2

                
            
            limits = limit_area[face]
            if limits is None:
                limits = [0,self.global_size,0,self.global_size]
            center = np.array([limits[0]+(limits[1]-limits[0])*np.random.random(),limits[2]+(limits[3]-limits[2])*np.random.random()])
            
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

            s = int(2**round(np.log2(np.floor(convert_pc_to_index(size, self.nres, self.size)/2)*2)))
            if force_size > 0:
                s = int(force_size)
            start_x = c_x - s // 2
            start_y = c_y - s // 2
            end_x = c_x + s // 2 + s%2
            end_y = c_y + s // 2 + s%2


            #If the random area is outside the sim
            if(start_x < 0 or start_y < 0 or end_x >= self.nres or end_y >= self.nres):
                continue
            cropped_cdens = c_dens[start_x:end_x, start_y:end_y]
            cropped_vdens = v_dens[start_x:end_x, start_y:end_y]

            #Verify if there is no low density region (outside cloud) inside the area
            if(((cropped_vdens < 10).sum()) > s*s*0.01):
                continue

            # Randomly choose a rotation (0, 90, 180, or 270 degrees)
            rotated_cdens = cropped_cdens
            rotated_vdens = cropped_vdens
            if random_rotate:
                k = np.random.choice([0, 1, 2, 3])
                rotated_cdens = np.rot90(cropped_cdens, k)
                rotated_vdens = np.rot90(cropped_vdens, k)


            b = (rotated_cdens, rotated_vdens)

            score = compute_img_score(b[0],b[1])
            if(np.random.random() > RANDOM_BATCH_SCORE_fct(score[0])):
                continue

            imgs.append(b)
            scores.append(score)
            areas_explored[face].append(center)
            img_generated += 1
        
        random_idx = np.random.permutation(len(imgs))
        r_imgs = []
        for r_id in random_idx:
            r_imgs.append(imgs[r_id])
        imgs = r_imgs
        r_scores = []
        for r_id in random_idx:
            r_scores.append(imgs[r_id])
        imgs = r_imgs
        scores = r_scores
        
        
        settings = {
            "SIM_name":self.name,
            "method": method.__name__,
            "img_number": len(imgs),
            "img_size": size,
            "areas_explored":str(areas_explored),
            "scores": str(scores),
            "scores_fct": inspect.getsourcelines(RANDOM_BATCH_SCORE_fct)[0][0],
            "scores_offset": str(RANDOM_BATCH_SCORE_offset),
            "number_goal": number,
            "iteration": iteration,
            "random_rotate": random_rotate,
        }
        return (imgs,settings)

    def plot(self,method=compute_column_density,plot_pdf=True,color_bar=False):
        """
        Plot simulations faces with probabiliy density function

        Args:
            method(function): Method to compute column density
        """
        column_density_xy = self._compute_c_density(method=method, axis=0)  # Top-down
        column_density_xz = self._compute_c_density(method=method, axis=1)  # Side view
        column_density_yz = self._compute_c_density(method=method, axis=2) # Front view
        

        fig, axes = plt.subplots(2 if plot_pdf else 1, 3, figsize=(12, 6 if plot_pdf else 3.5))
        if not(plot_pdf):
            axes = [axes]

        def _plot(column, data):
            cd = axes[0][column].imshow(data, extent=[self.axis[0][0], self.axis[0][1], self.axis[1][0],self.axis[1][1]], cmap="jet", norm=LogNorm(vmin=np.min(data), vmax=np.max(data)))
            pdf = compute_pdf(data)
            if plot_pdf:
                axes[1][column].plot([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
                axes[1][column].scatter([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
                axes[1][column].set_xlabel("s")
                axes[1][column].set_ylabel("p")
                axes[1][column].set_title("PDF")
            return cd, pdf

        # XY Projection (Top-down)
        _plot(0,column_density_xy)
        axes[0][0].set_title("Top-Down View (XY Projection)")
        axes[0][0].set_xlabel("X [pc]")
        axes[0][0].set_ylabel("Y [pc]")

        # XZ Projection (Side view)
        _plot(1,column_density_xz)

        axes[0][1].set_title("Side View (XZ Projection)")
        axes[0][1].set_xlabel("X [pc]")
        axes[0][1].set_ylabel("Z [pc]")

        # YZ Projection (Front view)
        cd, _ =  _plot(2,column_density_yz)
        axes[0][2].set_title("Front View (YZ Projection)")
        axes[0][2].set_xlabel("Y [pc]")
        axes[0][2].set_ylabel("Z [pc]")

        if color_bar:
            cbar = plt.colorbar(cd, ax=axes[0], orientation="vertical", fraction=0.02, pad=0.02)
            cbar.set_label("Column density ($cm^{-2}$)")

        return fig, axes

    def plot_correlation(self,method=compute_mass_weighted_density, axis=-1):
        """
        Plot correlation between the column density and the volumic density

        Args:
            method(function): Method to compute volumic density
            axis(int, default:-1): which face of the sim, if -1 all faces are taken
        """
        fig, ax = plt.subplots(1,1)
        if axis >= 0:
            column_density = np.log(self._compute_c_density(axis=axis).flatten())/np.log(10)
            volume_density = np.log(self._compute_v_density(method=method, axis=axis).flatten())/np.log(10)
        else:
            column_density = np.log(np.array([self._compute_c_density(axis=0),self._compute_c_density(axis=1),self._compute_c_density(axis=2)]).flatten())/np.log(10)
            volume_density = np.log(np.array([self._compute_v_density(method=method, axis=0),self._compute_v_density(method=method, axis=1),self._compute_v_density(method=method, axis=2)]).flatten())/np.log(10)

        _, _,_,hist = ax.hist2d(column_density, volume_density, bins=(256,256), norm=LogNorm())
        ax.set_xlabel(r"Column density ($log_{10}(cm^{-2})$)")
        ax.set_ylabel(r"Mass weighted density ($log_{10}(cm^{-3})$)")
        plt.colorbar(hist, ax=ax, label="counts")
        fig.tight_layout()
        return fig, ax