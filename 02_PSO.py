import random
import numpy as np

swarmsize = 20
alpha = 0.5 # proportion of velocity
beta = 1.5 # proportion of personal best
gamma = 1.5 # proportion of the informants
delta = 0.5 # proprtion of global best
epsilon = 1 #jump size of a particle
dimensions = 2 #dimensionality of the search space

target = np.ones(dimensions) # Target valuse for MSE

#class to store position, velocity and fitness for each particle
class Particle:
    def __init__(self):
        self.position = [random.uniform(-1, 1) for _ in range(dimensions)] #Randon initial position
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)] #Random initial velocity
        self.personalbest = self.position[:] #Initially set to the particle's starting position
        self.fitness = float('inf')
        self.informants = []

    #defining fitness function
    def fitness_function(self):
         position_array = np.array(self.position)
         mse = np.mean((position_array - target) ** 2)
         return mse
    
    # Defining personal best
    def personal_best(self):
        fitness = self.fitness_function()
        if fitness < self.fitness:
             self.personalbest = self.position[:]
             self.fitness = fitness
    
# Initializing swarm 
swarm = [Particle() for _ in range(swarmsize)]   
for particle in swarm:
    particle.informants = random.sample(swarm, k=min(5, swarmsize))

global_best_position = None
global_best_fitness = float('inf')

# PSO Optimization Loop
for iteration in range(100):
    for particle in swarm:
        # Calculating fitness for the current particle
        fitness = particle.fitness_function()
        # Checking if the current fitness is better than particle's personal best
        if fitness < particle.fitness:
            particle.personal_best()

        # Checking if the current fitness is better than the global best fitness
        if fitness < global_best_fitness:
            global_best_position = particle.position[:]
            global_best_fitness = fitness

    # Updating velocity and position for each particle
    for particle in swarm:
        personalBest = particle.personalbest
        global_best = global_best_position
        informant_best = min(particle.informants, key = lambda p:p.fitness).personalbest
        for i in range(dimensions): #Updating particle velocity based on personal, informant and global best
            b = random.uniform(0, beta) #Random coefficient for personal best
            c = random.uniform(0, gamma) #Random coefficient for informant's best
            d = random.uniform(0, delta) #Random coefficient  for global best
            #updating velocity based on personal, informant, and global best
            #vi ← αvi + b(x∗i − xi) + c(x+i − xi) + d(x!i − xi)
            particle.velocity[i] = (alpha* particle.velocity[i] + b * (personalBest[i] - particle.position[i])) + c * (informant_best[i] - particle.position[i] + d * (global_best[i] - particle.position[i]))

            # Updating position
            particle.position[i] += epsilon * particle.velocity[i] 

    if global_best_fitness < 0.05:
        break

print("Global Best Fitness: " , global_best_fitness)
print('Global Best Position: ', global_best_position)
    
