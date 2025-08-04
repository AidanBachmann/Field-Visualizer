# ---------- Imports ----------

import numpy as np
from numba import jit
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

# ---------- Function Definitions ----------

@jit(nopython = True)
def updateState(timeStep,d,w,dt,phi): # Update particle state using equations of motion
    # Rotating Multipole (circular trajectory)
    x = d*np.cos(w*timeStep*dt + phi) # Compute position
    y = d*np.sin(w*timeStep*dt + phi)

    vx = -d*w*np.sin(w*timeStep*dt + phi) # Compute velocity
    vy = d*w*np.cos(w*timeStep*dt + phi)

    ax = -d*(pow(w,2))*np.cos(w*timeStep*dt + phi) # Compute acceleration
    ay = -d*(pow(w,2))*np.sin(w*timeStep*dt + phi)

    # Antenna-like Trajectory (particles oscillate on x-axis)
    '''x = d*np.cos(w*timeStep*dt + phi) # Compute position
    y = 0

    vx = -d*w*np.sin(w*timeStep*dt + phi) # Compute velocity
    vy = 0

    ax = -d*(pow(w,2))*np.cos(w*timeStep*dt + phi) # Compute acceleration
    ay = 0'''

    return x,y,vx,vy,ax,ay


# Initialize simulation
def init(numParticles=4, # Number of particles in simulation
            numTimeSteps=20, # Number of time steps for simulation. 62 time steps for one complete cycle with omega = 1kHz.
            gridLen=1e6, # Grid length
            numGridPoints=21, # Num of grid points to cover grid length
            dt=0.0001, # Time step. dt = 0.0001 works well for w = 1e3, 0.00005 for high resolution.
            d=1, # Charge distance from origin
            w=1e3): # Angular frequnecy of charge rotation
            
    phi = initPhaseShift(numParticles) # Initialize phase shift of each particle
    particles,chargeSign = initParticles(numParticles,phi,d,w,dt,numTimeSteps) # Initialize all particle objects
    
    gridSpacing = gridLen/(numGridPoints-1) # Spacing of grid points as defined by length and number of points
    grid = np.zeros((numGridPoints,numGridPoints,numGridPoints,9,numTimeSteps)) # Initialize arrays to store all field information
    
    x = np.linspace(-(numGridPoints-1)/2,(numGridPoints-1)/2,numGridPoints,dtype='int')*gridSpacing # Initialize grid array
    gridx,gridy,gridz = np.meshgrid(x,x,x) # Generate meshgrid
    
    roll(gridx,3,int((numGridPoints-1)/2)) # Roll each meshgrid so each is centered around the origin
    roll(gridy,3,int((numGridPoints-1)/2))
    roll(gridz,3,int((numGridPoints-1)/2))

    return particles,chargeSign,grid,gridSpacing,gridx,gridy,gridz

def initParticles(numParticles,phi,d,w,dt,numTimeSteps):
    Q = 1 # Used for sign of charge (alternates for adjacent charges)
    particles = np.zeros([numParticles,numTimeSteps,6],dtype='float') # Array for particles. For all particles, at each time step store x,y,vx,vy,ax,ay
    chargeSign = np.zeros([numParticles])
    for i in np.linspace(0,numParticles-1,numParticles,dtype='int'):
        chargeSign[i] = Q
        Q *= -1 # Alternate sign of charge for arjacent charges
    computeTrajectories(numTimeSteps,numParticles,particles,phi,d,w,dt) # Compute particle trajectories according to EoM
    return particles,chargeSign
            
@jit(nopython = True)
def initPhaseShift(numParticles): # Initialize phase shift
    phi = np.ones(numParticles) # Array for initial phase angles of each particle
    for i in np.linspace(0,numParticles-1,numParticles).astype('int'):
        phi[i] = i*(2*np.pi/numParticles) # Compute phase
    return phi

def computeTrajectories(numTimeSteps,numParticles,particles,phi,d,w,dt): # Computes trajectories for all particles
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps,dtype='int'):
        for j in np.linspace(0,numParticles-1,numParticles,dtype='int'): 
            particles[j,i,0],particles[j,i,1],particles[j,i,2],particles[j,i,3],particles[j,i,4],particles[j,i,5] = updateState(i,d,w,dt,phi[j]) # Compute trajectory

@jit(nopython = True)
def computeDelayedTime(particle,x,y,z,timeStep,dt,c): # Compute delayed time for jth particle at nth time step
    R = np.array((x,y,z)) # Field point
    diff = np.zeros(3) # Separation vector
    t = timeStep*dt # Current time 

    # Error in delayed time should decrease as we turn back the clock before increasing again. So, if error begins to increase again,
    # then we know we have passed the relevant delayed time point. This decreases computation time for simulations with large Nt
    # since we must loop through a much smaller subset of time values.
    tridx = 0 # Index of delayed time
    if timeStep > 2:
        diff = R - np.array((particle[timeStep,0],particle[timeStep,1],0))
        preverror = abs(np.sqrt(np.dot(diff,diff)) - c*(t-(timeStep)*dt))
        for i in np.linspace(timeStep-1,0,timeStep-1).astype('int'):
            diff = R - np.array((particle[i,0],particle[i,1],0))
            error = abs(np.sqrt(np.dot(diff,diff)) - c*(t-i*dt))
            if error > preverror:
                tridx = i + 1 
                break
            else:
                preverror = error
    else:
        pass
    return tridx


'''@jit(nopython = True) # Old method of computing delayed time, way slower than other method and also not quite right
def computeDelayedTime(particle,x,y,z,timeStep,dt,c): #Compute delayed time for jth particle at nth time step
    R = np.array((x,y,z)) # Field point
    diff = np.zeros(3) # Separation vector
    t = timeStep*dt # Current time 

    error = np.zeros((timeStep))
    for i in np.linspace(timeStep-1,0,timeStep).astype('int'):
        diff = R - np.array((particle[i,0],particle[i,1],0))
        error[i] = abs(np.sqrt(np.dot(diff,diff)) - c*(t - i*dt))
    try:
        return np.where(error==min(error))[0][0]
    except:
        return 0'''

@jit(nopython = True)
def computeAllDelayedTime(particles,numParticles,numGridPoints,timeStep,dt,gridx,gridy,gridz,tr,c): # Compute delayed times for all particles at all time steps
    for p in np.linspace(0,numParticles-1,numParticles).astype('int'):
        for i in np.linspace(0,numGridPoints-1,numGridPoints).astype('int'):
            for j in np.linspace(0,numGridPoints-1,numGridPoints).astype('int'):
                for k in np.linspace(0,numGridPoints-1,numGridPoints).astype('int'):
                    tr[p,i,j,k] = computeDelayedTime(particles[p,:,:],gridx[i,j,k],gridy[i,j,k],gridz[i,j,k],timeStep,dt,c)
    return tr

@jit(nopython = True)
def computeParticleField(particle,x,y,z,tr,c,eCharge,Q,e0,u0): #Compute field from nth particle at some field point at jth time step
    R = np.array((x-particle[tr,0],y-particle[tr,1],z))
    if (np.dot(R,R) == 0.0):
        return np.zeros(9)
    RMagnitude = np.sqrt(np.dot(R,R))
    Rhat = R/RMagnitude
    v = np.array((particle[tr,2],particle[tr,3],0))
    a = np.array((particle[tr,4],particle[tr,5],0))
    u = c*Rhat - v

    E = (Q*eCharge/(4*np.pi*e0))*(RMagnitude/(pow(np.dot(R,u),3)))*((pow(c,2) - np.dot(v,v))*u + np.cross(R,np.cross(u,a)) )
    B = (1/c)*np.cross(Rhat,E)
    S = (1/u0)*np.cross(E,B)
    fields = np.array((E[0],E[1],E[2],B[0],B[1],B[2],S[0],S[1],S[2]))
    return fields

@jit(nopython = True)
def computeAllParticleFields(particles,chargeSign,numParticles,grid,numGridPoints,timeStep,dt,gridx,gridy,gridz,tr,c,eCharge,e0,u0): #Compute all particle contributions to all field points for jth time step
    # Formatting: 
    # particles = (numParticles,numTimeSteps,6) --> Stores x,y,vx,vy,ax,ay
    # grid = (numGridPoints,numGridPoints,numGridPoints,9,numTimeSteps) --> Stores Ex,Ey,Ez,Bx,By,Bz,Sx,Sy,Sz for every grid point at every time step
    tr = computeAllDelayedTime(particles,numParticles,numGridPoints,timeStep,dt,gridx,gridy,gridz,tr,c)
    #for i in tqdm(np.linspace(0,numGridPoints-1,numGridPoints,dtype='int'),total=numGridPoints): # Uncomment this line to get a progress bar, not compatible with jit
    for i in np.linspace(0,numGridPoints-1,numGridPoints).astype('int'):
        for j in np.linspace(0,numGridPoints-1,numGridPoints).astype('int'):
            for k in np.linspace(0,numGridPoints-1,numGridPoints).astype('int'):
                for p in np.linspace(0,len(particles)-1,len(particles)).astype('int'):
                    fields = computeParticleField(particles[p,:,:],gridx[i,j,k],gridy[i,j,k],gridz[i,j,k],tr[p,i,j,k],c,eCharge,chargeSign[p],e0,u0)
                    for dim in np.linspace(0,2,3).astype('int'):
                        grid[i,j,k,dim,timeStep] += fields[dim]
                        grid[i,j,k,dim+3,timeStep] += fields[dim+3]
                        grid[i,j,k,dim+6,timeStep] += fields[dim+6]

@jit(nopython = True)
def computeFieldsForNTimeSteps(particles,chargeSign,numParticles,grid,numGridPoints,dt,numTimeSteps,gridx,gridy,gridz,tr,c,eCharge,e0,u0): #Compute all field contributions for all time steps
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps).astype('int'):
        computeAllParticleFields(particles,chargeSign,numParticles,grid,numGridPoints,i,dt,gridx,gridy,gridz,tr,c,eCharge,e0,u0)
        print(f'Completed time step {i+1} of {numTimeSteps}.')

def plotParticlePos(particle,timeStep,particleNum,color):
    plt.scatter(particle[timeStep,0],particle[timeStep,1],color=color)
    plt.text(particle[timeStep,0],particle[timeStep,1],"P%s T%s" % (particleNum,timeStep))

def plotParticleTrajectory(particles,numTimeSteps,numParticles):
    colors = plt.cm.hsv(np.linspace(0,1,numParticles)) # Create unique color for each particle
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps,dtype='int'):
        plt.cla()
        for j in np.linspace(0,numParticles-1,numParticles,dtype='int'):
            plt.scatter(particles[j,0:i,0],particles[j,0:i,1],color=colors[j])
            for idx in np.linspace(0,i,i,dtype='int'):
                plt.text(particles[j,idx,0],particles[j,idx,1],"P%s T%s" % (j,idx))
        plt.grid()
        plt.savefig('z_Trajectories/Trajectories_'+'{:05d}'.format(i+1)+'.png') # Save figure
    plt.savefig('z_Trajectories/Trajectory.jpg')
    makeGif(numTimeSteps,'Trajectories') # Make gif
    os.chdir('z_Trajectories/')
    cleanAll()
    os.chdir('../')
    plt.clf()
        
def plot2DE(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy): # Plot 2D Electric field
    colors = plt.cm.hsv(np.linspace(0,1,numParticles)) # Create unique color for each particle
    plt.figure(1)
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps,dtype='int'):
        plt.quiver(gridx[:,:,int((numGridPoints-1)/2)],gridy[:,:,int((numGridPoints-1)/2)],grid[:,:,0,0,i],grid[:,:,0,1,i])
        for j in np.linspace(0,numParticles-1,numParticles,dtype='int'):
            plotParticlePos(particles[j,:,:],i,j,colors[j])
        plt.savefig('z_E/E_'+'{:05d}'.format(i+1)+'.png') # Save figure
        plt.cla() # Clear axes

def plot2DB(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy): # Plot 2D Magnetic field
    colors = plt.cm.hsv(np.linspace(0,1,numParticles)) # Create unique color for each particle
    plt.figure(2)
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps,dtype='int'):
        plt.quiver(gridx[:,:,int((numGridPoints-1)/2)],gridy[:,:,int((numGridPoints-1)/2)],grid[:,:,0,3,i],grid[:,:,0,4,i])
        for j in np.linspace(0,numParticles-1,numParticles,dtype='int'):
            plotParticlePos(particles[j,:,:],i,j,colors[j])
        plt.savefig('z_B/B_'+'{:05d}'.format(i+1)+'.png') # Save figure
        plt.cla() # Clear axes

def plot2D(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,zsliceIdx,makeGifs,clearFigures):
    colors = plt.cm.hsv(np.linspace(0,1,numParticles)) # Create unique color for each particle
    fig,ax = plt.subplots(1,2,figsize=(20,11))
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps,dtype='int'):
        ax[0].set_title(f'Electric Field at z = {gridx[0,0,zsliceIdx]}')
        ax[1].set_title(f'Magnetic Field at z = {gridx[0,0,zsliceIdx]}')
        ax[0].quiver(gridx[:,:,zsliceIdx],gridy[:,:,zsliceIdx],grid[:,:,zsliceIdx,0,i],grid[:,:,zsliceIdx,1,i])
        ax[1].quiver(gridx[:,:,zsliceIdx],gridy[:,:,zsliceIdx],grid[:,:,zsliceIdx,3,i],grid[:,:,zsliceIdx,4,i])
        for j in np.linspace(0,numParticles-1,numParticles,dtype='int'):
            ax[0].scatter(particles[j,i,0],particles[j,i,1],color=colors[j])
            ax[0].text(particles[j,i,0],particles[j,i,1],"P%s T%s" % (j,i))
            ax[1].scatter(particles[j,i,0],particles[j,i,1],color=colors[j])
            ax[1].text(particles[j,i,0],particles[j,i,1],"P%s T%s" % (j,i))
        plt.savefig('z_Fields/Fields_'+'{:05d}'.format(i+1)+'.png') # Save figure
        ax[0].cla() # Clear axes
        ax[1].cla()
    if makeGifs:
        makeGif(numTimeSteps,'Fields') # Make gif
    if clearFigures:
        os.chdir('z_Fields/')
        cleanAll()
        os.chdir('../')

def plot2DFieldNorms(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,zsliceIdx,makeGifs,clearFigures):
    colors = plt.cm.hsv(np.linspace(0,1,numParticles)) # Create unique color for each particle
    fig,ax = plt.subplots(1,2,figsize=(20,11))
    Enorm = np.sqrt(pow(grid[:,:,zsliceIdx,0,:],2) + pow(grid[:,:,zsliceIdx,1,:],2) + pow(grid[:,:,zsliceIdx,2,:],2))
    Bnorm = np.sqrt(pow(grid[:,:,zsliceIdx,3,:],2) + pow(grid[:,:,zsliceIdx,4,:],2) + pow(grid[:,:,zsliceIdx,5,:],2))
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps,dtype='int'):
        ax[0].set_title(f'Electric Field Magnitude at z = {gridx[0,0,zsliceIdx]}')
        ax[1].set_title(f'Magnetic Field Magnitude at z = {gridx[0,0,zsliceIdx]}')
        ax[0].pcolormesh(gridx[:,:,zsliceIdx],gridy[:,:,zsliceIdx],Enorm[:,:,i],shading='gouraud')
        ax[1].pcolormesh(gridx[:,:,zsliceIdx],gridy[:,:,zsliceIdx],Bnorm[:,:,i],shading='gouraud')
        for j in np.linspace(0,numParticles-1,numParticles,dtype='int'):
            ax[0].scatter(particles[j,i,0],particles[j,i,1],color=colors[j])
            ax[0].text(particles[j,i,0],particles[j,i,1],"P%s T%s" % (j,i))
            ax[1].scatter(particles[j,i,0],particles[j,i,1],color=colors[j])
            ax[1].text(particles[j,i,0],particles[j,i,1],"P%s T%s" % (j,i))
        plt.savefig('z_FieldNorms/FieldNorms_'+'{:05d}'.format(i+1)+'.png') # Save figure
        ax[0].cla() # Clear axes
        ax[1].cla()
    if makeGifs:
        makeGif(numTimeSteps,'FieldNorms') # Make gif
    if clearFigures:
        os.chdir('z_FieldNorms/')
        cleanAll()
        os.chdir('../')

def plot2DS(particles,numParticles,numTimeSteps,numGridPoints,grid,gridx,gridy,zsliceIdx,makeGifs,clearFigures): # Plot 2D Poynting field
    colors = plt.cm.hsv(np.linspace(0,1,numParticles)) # Create unique color for each particle
    fieldNorm = np.sqrt(pow(grid[:,:,zsliceIdx,6,:],2) + pow(grid[:,:,zsliceIdx,7,:],2)) # Magnitude of Poynting field
    for i in np.linspace(0,numTimeSteps-1,numTimeSteps,dtype='int'):
        plt.title(f'Magnitude of Poynting Vector at z = {gridx[0,0,zsliceIdx]}')
        plt.pcolormesh(gridx[:,:,zsliceIdx],gridy[:,:,zsliceIdx],fieldNorm[:,:,i],shading='gouraud')
        for j in np.linspace(0,numParticles-1,numParticles,dtype='int'):
            plotParticlePos(particles[j,:,:],i,j,colors[j])
        plt.savefig('z_S/S_'+'{:05d}'.format(i+1)+'.png') # Save figure
        plt.cla() # Clear axes
    if makeGifs:
        makeGif(numTimeSteps,'S') # Make gif
    if clearFigures:
        os.chdir('z_S/')
        cleanAll()
        os.chdir('../')

def roll(array,numAxes,shift): # Roll axes to origin
    for i in np.linspace(0,numAxes-1,numAxes,dtype='int'):
        array = np.roll(array,shift,axis=i)

def makeGif(Nt,title): # Make gif from frames
    frames = []
    for i in np.linspace(1,Nt,Nt,dtype='int'):
        frames.append(Image.open(f'z_{title}/{title}_'+'{:05d}'.format(i)+'.png'))
    frame_one = frames[0]
    frame_one.save(f'z_{title}/{title}.gif', format="GIF", append_images=frames,
               save_all=True, duration=30, loop=0)
    
def cleanAll(): # Remove all figures
    for file in os.listdir(os.getcwd()):
        if file.endswith('.png'):
            os.remove(file) 