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
        self.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"sims/"+name+"/")
        """Path to the folder where the simulation is stored"""
        self.file = os.path.join(self.folder,SIM_DATA_NAME)
        """Path to the simulation data"""
        self.data = None
        """Raw simulation data"""
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

    def init(self):
        """
        Load files and data in self variables
        """
        simfile = fits.open(self.file)
        self.data = simfile[0].data
        simfile.close()

        with open(os.path.join(self.folder,"processing_config.json"), "r") as file:
            self.header = json.load(file)
        self.nres = self.header
        self.nres = self.header["run_parameters"]["nres"]
        self.relative_size = self.header["run_parameters"]["size"]
        self.center = np.array([self.header["run_parameters"]["xcenter"],self.header["run_parameters"]["ycenter"],self.header["run_parameters"]["zcenter"]])
        self.cell_size = (self.global_size*self.relative_size/self.nres) * u.parsec
        self.cell_size = self.cell_size.to(u.cm)
        self.size = self.global_size*self.relative_size
        self.axis = ([self.center[0]*self.global_size-self.size/2,self.center[0]*self.global_size+self.size/2],[self.center[1]*self.global_size-self.size/2,self.center[1]*self.global_size+self.size/2],[self.center[2]*self.global_size-self.size/2,self.center[2]*self.global_size+self.size/2])    

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
                print("Error: failed to generated all the random batches, nbr of img generated:"+str(len(imgs)))
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
            c_x = convert_pc_to_index(c_x)-int(np.floor(self.axis[0][0]/self.size*self.nres))
            c_y = convert_pc_to_index(c_y)-int(np.floor(self.axis[0][0]/self.size*self.nres))

            s = int(2**round(np.log2(np.floor(convert_pc_to_index(size)/2)*2)))
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

    def plot(self,method=compute_column_density):
        """
        Plot simulations faces with probabiliy density function

        Args:
            method(function): Method to compute column density
        """
        column_density_xy = self._compute_c_density(method=method, axis=0)  # Top-down
        column_density_xz = self._compute_c_density(method=method, axis=1)  # Side view
        column_density_yz = self._compute_c_density(method=method, axis=2) # Front view
        _, axes = plt.subplots(2, 3, figsize=(9, 6))

        def _plot(column, data):
            cd = axes[0][column].imshow(data, extent=[self.axis[0][0], self.axis[0][1], self.axis[1][0],self.axis[1][1]], cmap="jet", norm=LogNorm(vmin=np.min(data), vmax=np.max(data)))
            plt.colorbar(cd,ax=axes[0][column], label=method.__name__)
            pdf = compute_pdf(data)
            axes[1][column].plot([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
            axes[1][column].scatter([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
            axes[1][column].set_xlabel("s")
            axes[1][column].set_ylabel("p")
            axes[1][column].set_title("PDF")

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
        _plot(2,column_density_yz)
        axes[0][2].set_title("Front View (YZ Projection)")
        axes[0][2].set_xlabel("Y [pc]")
        axes[0][2].set_ylabel("Z [pc]")

    def plot_correlation(self,method=compute_volume_weighted_density, axis=0):
        """
        Plot correlation between the column density and the volumic density

        Args:
            method(function): Method to compute volumic density
            axis(int, default:0): which face of the sim
        """
        fig, ax = plt.subplots(1,1)
        column_density = np.log(self._compute_c_density(axis=axis).flatten())/np.log(10)
        volume_density = np.log(self._compute_v_density(method=method, axis=axis).flatten())/np.log(10)
        _, _,_,hist = ax.hist2d(column_density, volume_density, bins=(256,256), norm=LogNorm())
        plt.colorbar(hist, ax=ax)
        fig.tight_layout()