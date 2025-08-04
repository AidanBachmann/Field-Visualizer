# ---------- Imports ----------

import Functions as funcs
import numpy as np
import time
from numba import jit

# ---------- Simulation Parameters ----------

d = 1 # Charge distance from origin
w = 1e3 # Angular frequnecy of charge rotation
eCharge = 1.6e-19 # Charge of each particle
e0 = 8.85e-8 # Permitivitty of free space, use this to set light speed (effectively change length scale of simulation), free space = 8.85e-12. To slow down, increase value.
u0 = 1.26e-6 # Permiability of free space, free space = 1.26e-12
c = 1/np.sqrt(e0*u0) # Speed of light
gridLen = int(1e4) # Grid length
numGridPoints = int(21) # Num of grid points to cover grid length
zsliceIdx = -1 # Index of z plane to plot fields, (numGridPoints-1)/2 is z = 0
numParticles = int(8) # Number of particles in simulation
dt = 0.00005 # Time step size
numTimeSteps = 200 # Number of time steps to simulate, 200
propFrac = c*numTimeSteps*dt/(2*gridLen) # Fraction of grid E propagates over during the simulation

plotTrajectories = True # Plot particle trajectories
plotFields = True # Plot electric and magnetic fields
plotS = True # Plot Poynting vector field
makeGifs = True # Make gifs from figures
clearFigures = True # Deletes all figures (not gifs) at the end of the run

# ---------- Main ----------

particles,chargeSign,grid,gridSpacing,gridx,gridy,gridz = funcs.init(numParticles,numTimeSteps,gridLen,numGridPoints,dt,d,w) # Initialize simulation                   
tr = np.zeros((numParticles,numGridPoints,numGridPoints,numGridPoints)).astype('int') # Delayed time array
print(f'Computing fields for Nt = {numTimeSteps} time steps.')
start = time.time()
funcs.computeFieldsForNTimeSteps(particles,chargeSign,numParticles,grid,numGridPoints,dt,numTimeSteps,gridx,gridy,gridz,tr,c,eCharge,e0,u0)
end = time.time()
print(f'Simulation completed in {end - start} seconds.')

if plotTrajectories:
    funcs.plotParticleTrajectory(particles,numTimeSteps,numParticles)
if plotFields:
    funcs.plot2D(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,zsliceIdx,makeGifs,clearFigures)
    print('Finished animation of E and B quiver plots.')
    funcs.plot2DFieldNorms(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,zsliceIdx,makeGifs,clearFigures)
    print('Finished animation of E and B field magnitude plots.')
if plotS:
    funcs.plot2DS(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,zsliceIdx,makeGifs,clearFigures)
    print('Finished animation of Poynting vector quiver plot.')