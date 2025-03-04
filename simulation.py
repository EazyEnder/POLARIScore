from utils import *
from config import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_sim(method=compute_column_density):
    column_density_xy = method(DATA, axis=0)  # Top-down
    column_density_xz = method(DATA, axis=1)  # Side view
    column_density_yz = method(DATA, axis=2)  # Front view
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    def plot(column, data):
        cd = axes[0][column].imshow(data, extent=[SIM_axis[0][0], SIM_axis[0][1], SIM_axis[1][0],SIM_axis[1][1]], cmap="jet", norm=LogNorm(vmin=np.min(data), vmax=np.max(data)))
        plt.colorbar(cd,ax=axes[0][column], label=method.__name__)
        pdf = compute_pdf(data)
        axes[1][column].plot([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
        axes[1][column].scatter([(pdf[1][i+1]+pdf[1][i])/2 for i in range(len(pdf[1])-1)],pdf[0])
        axes[1][column].set_xlabel("s")
        axes[1][column].set_ylabel("p")
        axes[1][column].set_title("PDF")

    # XY Projection (Top-down)
    plot(0,column_density_xy)
    axes[0][0].set_title("Top-Down View (XY Projection)")
    axes[0][0].set_xlabel("X [pc]")
    axes[0][0].set_ylabel("Y [pc]")

    # XZ Projection (Side view)
    plot(1,column_density_xz)

    axes[0][1].set_title("Side View (XZ Projection)")
    axes[0][1].set_xlabel("X [pc]")
    axes[0][1].set_ylabel("Z [pc]")

    # YZ Projection (Front view)
    plot(2,column_density_yz)
    axes[0][2].set_title("Front View (YZ Projection)")
    axes[0][2].set_xlabel("Y [pc]")
    axes[0][2].set_ylabel("Z [pc]")

def plot_correlation(method=compute_volume_weighted_density, axis=0):
    fig, ax = plt.subplots(1,1)
    column_density = np.log(compute_column_density(DATA, axis).flatten())/np.log(10)
    volume_density = np.log(method(DATA, axis).flatten())/np.log(10)
    _, _,_,hist = ax.hist2d(column_density, volume_density, bins=(256,256), norm=LogNorm())
    plt.colorbar(hist, ax=ax)
    fig.tight_layout()

if __name__ == "__main__":
    plot_sim(method=compute_mass_weighted_density)
    plt.show()