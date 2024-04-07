import torch
from torch import optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def ackley(x, y):
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))) + torch.exp(torch.tensor([1.0])) + 20


###############################################################################################
# Plot the objective function

# You will need to use Matplotlib's 3D plotting capabilities to plot the objective functions.
# Alternate plotting libraries are acceptable.
###############################################################################################


def plot_function(name: str, x, y, z):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(plt.LinearLocator(10))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # save the plot
    plt.savefig(name)


###############################################################################################
# STOCHASTIC GRADIENT DESCENT

# Initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# Use Stochastic Gradient Descent in Pytorch to optimize the objective function.

# Saev the values of the objective function over 5000 iterations in a list.

# Print the values of x, y, and the objective function after optimization.
###############################################################################################

'''


YOUR CODE HERE


'''





###############################################################################################
# Adam Optimizer

# Re-initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# Use the Adam optimizer in Pytorch to optimize the objective function.

# Saev the values of the objective function over 5000 iterations in a list.

# Print the values of x, y, and the objective function after optimization.
###############################################################################################

'''


YOUR CODE HERE


'''


###############################################################################################
# Comparing convergence rates

# Plot the previously stored values of the objective function over 5000 iterations for both SGD and Adam in a single plot.
###############################################################################################

'''


YOUR CODE HERE


'''
