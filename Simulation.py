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
numParticles = int(8) # Number of particles in simulation
#dt = min(d*((2*np.pi)/numParticles)/c,2*np.pi/w*0.01) # Define the time step as 1/100 of a period
#numTimeSteps = int((2*np.pi/w)/dt) # Number of time steps for simulation
dt = 0.0001
numTimeSteps = 100
propFrac = c*numTimeSteps*dt/(2*gridLen) # Fraction of grid E propagates over during the simulation

useDeafultEOM = True # Use default equations of motion, define in Functions.py
plotTrajectories = True # Plot particle trajectories
plotFields = True # Plot electric and magnetic fields
plotS = True # Plot Poynting vector field
makeGifs = True # Make gifs from figures
clearFigures = True # Deletes all figures (not gifs) at the end of the run

# ---------- Define Equations of Motion ----------

if useDeafultEOM == True: # Use default equations of motion (circular trajectory)
    print('Using default equations of motion.')
    fpos = funcs.computePosDefault
    fvel = funcs.computeVelDefault
    faccel = funcs.computeAccelDefault
else:
    print('Using user-defined equations of motion.')
    # Define alternative equations of motion
    @jit(nopython = True)
    def computePos(timeStep,pos,d,w,dt,phi):
        pos[0,timeStep] = d*np.cos(w*timeStep*dt + phi)
        pos[1,timeStep] = d*np.sin(w*timeStep*dt + phi)

    @jit(nopython = True)    
    def computeVel(timeStep,vel,d,w,dt,phi):
        vel[0,timeStep] = -d*w*np.sin(w*timeStep*dt + phi)
        vel[1,timeStep] = d*w*np.cos(w*timeStep*dt + phi)

    @jit(nopython = True)
    def computeAccel(timeStep,accel,d,w,dt,phi):
        accel[0,timeStep] = -d*(w**2)*np.cos(w*timeStep*dt + phi)
        accel[1,timeStep] = -d*(w**2)*np.sin(w*timeStep*dt + phi)

    fpos = computePos
    fvel = computeVel
    faccel = computeAccel

# ---------- Main ----------

particles,chargeSign,grid,gridSpacing,gridx,gridy,gridz = funcs.init(fpos,fvel,faccel, # Initialize simulation
                                                    numParticles,numTimeSteps,gridLen,numGridPoints,dt,d,w)
tr = np.zeros((numParticles,numGridPoints,numGridPoints,numGridPoints)).astype('int') # Delayed time array
print(f'Computing fields for Nt = {numTimeSteps} time steps.')
start = time.time()
funcs.computeFieldsForNTimeSteps(particles,chargeSign,numParticles,grid,numGridPoints,dt,numTimeSteps,gridx,gridy,gridz,tr,c,eCharge,e0,u0)
end = time.time()
print(f'Simulation completed in {end - start} seconds.')

#if plotTrajectories:
#    funcs.plotParticleTrajectory(particles,numTimeSteps,numParticles)
if plotFields:
    funcs.plot2D(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,makeGifs,clearFigures)
    #funcs.plot2DFieldNorms(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,makeGifs,clearFigures)
if plotS:
    funcs.plot2DS(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,makeGifs,clearFigures)