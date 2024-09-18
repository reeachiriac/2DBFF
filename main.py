import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys
from colorama import Fore

# checking that right number of arguments has been given
if len(sys.argv) < 4:
    print('\n', Fore.RED + "Not enough arguments given", '\n')
    sys.exit(1)
elif len(sys.argv) > 5:
    print('\n', Fore.RED + "Too many arguments given", '\n')
    sys.exit(1)

# The arguments:
# - n (n x n domain);
# - epsilon (the spacing) - we will not allow more than 3 decimals;
# - m: as defined in the paper, 2mx2m is the dimension of our truncation square;
# - threshold(optional)
n = float(sys.argv[1])
if n < 0:
    n = -n
elif n == 0:
    print(Fore.RED + "Domain cannot be null.")
epsilon = round(float(sys.argv[2]), 3)
if epsilon == 0:
    print(Fore.RED + "Mesh value too small.")
    sys.exit()
m = int(sys.argv[3])

# We keep track of how long our code takes to run
startTime = time.time()

# We find the square root of the number of point on our the grid, nr
nr = int(n / epsilon)

# We slightly refactor n
n = nr * epsilon

# calculate the bound, bnd
bnd = epsilon * int(m / epsilon)

# generating once and then storing our values for (alpha)_{i,j}'s
r = nr + 2 * int(m / epsilon)
alpha = [[np.random.normal(0,2) for _ in range(r+1)] for _ in range(r+1)]


# We define the function
def xa(a1, a2):
    # Initialize the sum
    xa = 0
    # Perform the double summation with respect to the truncation as a 4x4 square
    for i in np.arange(a1-bnd, a1+bnd+0.001, epsilon):
        for j in np.arange(a2-bnd, a2+bnd+0.001, epsilon):
            iAlph = i / epsilon + int(m / epsilon)
            jAlph = j / epsilon + int(m / epsilon)
            xa += alpha[int(iAlph)][int(jAlph)] * np.exp(-((a1-i) ** 2 + (a2-j) ** 2))
    xa = xa * math.sqrt(2 / math.pi)
    return xa


# Generate grid
xs = np.linspace(0, n, nr)
ys = np.linspace(0, n, nr)

# To each point (x,y), we associate a value (our 'height' function)
z = []
for i in xs:
    for j in ys:
        z = np.append(z, xa(i, j))
# We rearrange z as a matrix
Z = z.reshape(int(nr), int(nr))

# plotting the field w.r.t. whether a threshold was provided or not
if len(sys.argv) == 4:
    plt.contourf(xs, ys, Z, cmap='coolwarm')
    plt.colorbar()
elif len(sys.argv) == 5:
    threshold = float(sys.argv[4])
    custom_levels = [-100, threshold, 100]
    plt.contourf(xs, ys, Z, levels = custom_levels, colors=['aliceblue', 'seagreen'])
    plt.colorbar(ticks = custom_levels)

endTime = time.time()
elapsedTime = endTime - startTime
print('\n', Fore.GREEN + 'Elapsed time: ', elapsedTime, '\n' + Fore.WHITE)

plt.title('2-D Bargmann-Fock Field with mesh ' + str(epsilon))
plt.show()
