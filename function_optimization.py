import torch
from torch import optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import AutoLocator, LinearLocator
from tqdm import tqdm

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.get_device_name(0))

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


def plot_function(function, save_name, start_input, end_input):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data.
    X = torch.linspace(start_input, end_input, 100)
    Y = torch.linspace(start_input, end_input, 100)
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    Z = function(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # # Customize the z axis.
    # ax.zaxis.set_major_locator(AutoLocator())
    # # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Set the title
    ax.set_title(save_name)

    # Save the figure
    plt.savefig(save_name + '.png')
    
#plot the rosenbrock function
plot_function(rosenbrock, 'rosenbrock [-30,30]', -30, 30)
#clear the plot
plt.clf()

#plot the beale function
plot_function(beale, 'beale [-30,30]', -30, 30)
#clear the plot
plt.clf()

#plot the ackley function
plot_function(ackley, 'ackley [-30,30]', -30, 30)
#clear the plot
plt.clf()

#plot the ackley function in [-3,3]
plot_function(ackley, 'ackley [-3,3]', -3, 3)
#clear the plot
plt.clf()

###############################################################################################
# STOCHASTIC GRADIENT DESCENT

# Initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# Use Stochastic Gradient Descent in Pytorch to optimize the objective function.

# Save the values of the objective function over 5000 iterations in a list.

# Print the values of x, y, and the objective function after optimization.
###############################################################################################
    
def SGD(function, lr=0.0015, n_iters=5000):
    x = torch.tensor([10.0], dtype=torch.float, requires_grad=True)
    y = torch.tensor([10.0], dtype=torch.float, requires_grad=True)
    optimizer = optim.SGD([x, y], lr=lr)
    values = []
    
    #implement the optimization ref: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        loss = function(x, y)
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_value_([x, y], clip_value=10)
        optimizer.step()
        values.append(loss.item())
    return x, y, values

#optimize the rosenbrock function
x, y, values_rosenbrock_SGD = SGD(rosenbrock)

#print the values of x, y, and the objective function after optimization
print('Rosenbrock function')
print('x:', x.item())
print('y:', y.item())
print('Objective function:', values_rosenbrock_SGD[-1])

#optimize the beale function
x, y, values_beale_SGD = SGD(beale)

#print the values of x, y, and the objective function after optimization
print('Beale function')
print('x:', x.item())
print('y:', y.item())
print('Objective function:', values_beale_SGD[-1])

#optimize the ackley function
x, y, values_ackley_SGD = SGD(ackley)

#print the values of x, y, and the objective function after optimization
print('Ackley function')
print('x:', x.item())
print('y:', y.item())
print('Objective function:', values_ackley_SGD[-1])

###############################################################################################
# Adam Optimizer

# Re-initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# Use the Adam optimizer in Pytorch to optimize the objective function.

# Saev the values of the objective function over 5000 iterations in a list.

# Print the values of x, y, and the objective function after optimization.
###############################################################################################

def adam_optim(function, lr=0.0015, n_iters=5000):
    x = torch.tensor([10.0], requires_grad=True)
    y = torch.tensor([10.0], requires_grad=True)
    optimizer = optim.Adam([x, y], lr=lr)
    values = []
    #implement the optimization ref: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        loss = function(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_value_([x, y], clip_value=10)
        optimizer.step()
        values.append(loss.item())
    return x, y, values

#optimize the rosenbrock function
x, y, values_rosenbrock_adam = adam_optim(rosenbrock)

#print the values of x, y, and the objective function after optimization
print('Rosenbrock function')
print('x:', x.item())
print('y:', y.item())
print('Objective function:', rosenbrock(x, y).item())

#optimize the beale function
x, y, values_beale_adam = adam_optim(beale)

#print the values of x, y, and the objective function after optimization
print('Beale function')
print('x:', x.item())
print('y:', y.item())
print('Objective function:', beale(x, y).item())

#optimize the ackley function
x, y, values_ackley_adam = adam_optim(ackley)

#print the values of x, y, and the objective function after optimization
print('Ackley function')
print('x:', x.item())
print('y:', y.item())
print('Objective function:', ackley(x, y).item())


###############################################################################################
# Comparing convergence rates

# Plot the previously stored values of the objective function over 5000 iterations for both SGD and Adam in a single plot.
###############################################################################################

#plot the saved values of the objective function over the iterations
def plot_values(values, title):
    plt.plot(values[0], label='SGD')
    plt.plot(values[1], label='Adam')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Objective function')
    plt.title(title)
    plt.savefig(title + '.png')
    plt.clf()
    
#plot rosenbrock function values
plot_values([values_rosenbrock_SGD, values_rosenbrock_adam], 'Rosenbrock SGD vs Adam convergence rates')

#plot beale function values
plot_values([values_beale_SGD, values_beale_adam], 'Beale SGD vs Adam convergence rates')

#plot ackley function values
plot_values([values_ackley_SGD, values_ackley_adam], 'Ackley SGD vs Adam convergence rates')