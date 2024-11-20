import numpy as np
import random

#sigmoid activation function and  it's derivation
def sigmoid(x):
    return 1/(1+ np.exp(-x))

#relu activation function and it's derivation
def relu(x): 
    return np.maximum(0,x)

#tanh activation function and it's derivation
def tanh(x):  
      return np.tanh(x)

##Defining the architecture
layer_sizes = [5, 3, 3, 1] #5 input features, 3 nodes in 1st hidden layers, 3 nodes in 2nd hidden layer, 1 output
activation_functions = [sigmoid, tanh, relu]   #Activation Functions for each layer sigmoid for 1st hidden layer, tanh for 2nd hidden layer and relu for output layer

# Initializing ANN
def initialize_ANN(layer_sizes):   
        weights = []         #initializes an empty list where each item will store the weight
        biases = []          #initializes an empty list where each item will store the biases
        
        for i in range(len(layer_sizes) -1):          #This loop excludes the input layer as weights & biases connect layers to one another
            weights.append(np.random.rand(layer_sizes[i], layer_sizes[i+1]) *0.1)       #initializes weight between layer i & layer i+1 using random values & multiplying by 0.1 makes the initial bias value down
            biases.append(np.random.rand(layer_sizes[i+1])* 0.1)                        #initializes the bias in layer i+1 with random values
        return weights, biases

# Initializing Feed-forward
def feed_forward(X, weights, biases, activation_functions):
        for i in range(len(weights)):
             a = np.dot(X, weights[i]) + biases[i]  #calculates the input data X with weight of the currentlayer and ten adds the bias
             X = activation_functions[i](a)  #Applies activation function for the current layer 
        return X

# Defining Neural Networks for PSO
def neural_network(X, weights, biases, activation_functions):
      return feed_forward(X, weights, biases, activation_functions)

# Defining PSO
swarmsize = 20
alpha = 0.5 # proportion of velocity
beta = 1.5 # proportion of personal best
gamma = 1.5 # proportion of the informants
delta = 0.5 # proprtion of global best
epsilon = 1 #jump size of a particle
dimensions = sum(layer_sizes[i]*layer_sizes[i+1]
                 for i in range(len(layer_sizes)-1)) + sum(layer_sizes[1:])

# Target value for minimization
target = 0

#class to store position, velocity and fitness for each particle
class Particle:
    def __init__(self):
        self.position = [random.uniform(-1, 1) for _ in range(dimensions)] #Randon initial position
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)] #Random initial velocity
        self.personalbest = self.position[:] #Initially set to the particle's starting position
        self.fitness = float('inf')
        self.informants = []

     # Initializing Feed-forward
    def feed_forward(self, X, weights, biases, activation_functions):
        for i in range(len(weights)):
             a = np.dot(X, weights[i]) + biases[i]  #calculates the input data X with weight of the currentlayer and ten adds the bias
             X = activation_functions[i](a)  #Applies activation function for the current layer 
        return X

    def fitness_function(self, layer_sizes, activation_functions):
         # Generating synthetic input data
         X = np.random.randn(10, layer_sizes[0])

         #Decoding position to weights and biases
         weights, biases = self.decode_position(layer_sizes)

         # Getting network predictions
         predictions = self.feed_forward(X, weights, biases, activation_functions)

         # Calculating the Mean Squared Error with respect to the target value
         mse = np.mean((predictions - target) **2)
         return mse
    
    def decode_position(self, layer_sizes):
         # Intialize empty lists to store weights and biases
         weights, biases, idx = [], [], 0

         for i in range(len(layer_sizes) - 1):
              # Calculates the number of weights for the current layer
              weight_size = layer_sizes[i] * layer_sizes[i + 1]
              # Calculates thw number of biases for the current layer
              bias_size = layer_sizes[i + 1]
              
              # Extracting and reshaping the the weights for the current layer
              layer_weights = np.array(self.position[idx:idx + weight_size]).reshape(layer_sizes[i], layer_sizes[i+1])
              weights.append(layer_weights)
              # Updating the index pointer by the number of weights in the current layer
              idx += weight_size
              # Extracting the biases from the position vector for the next layer
              layer_biases = np.array(self.position[idx:idx + bias_size])
              biases.append(layer_biases)
              # Updating the index pointer
              idx += bias_size

         return weights, biases
    
# Initializing the swarm and assigning the informants
swarm = [Particle() for _ in range(swarmsize)]
for particle in swarm:
     particle.informants = random.sample(swarm, k=min(5, swarmsize))


global_best_position = None
global_best_fitness = float('inf')


# PSO Optimization loop
for iteration in range(100):
     for particle in swarm:
          # Calculating the fitness of the current particle
          fitness = particle.fitness_function(layer_sizes, activation_functions)
          # Checking if teh current fitness is better then the particle's personal best fitness
          if fitness < particle.fitness:
               # Updating teh particle's personal best position and fitness
               particle.personal_best_position = particle.position[:]
               particle.personal_best_fitness = fitness
          # Checking if the current fitness is better than the global best fitness
          if fitness < global_best_fitness:
               # Updating teh global best position and fitness
               global_best_position = particle.position[:]
               global_best_fitness = fitness

     # Updating velocity and position for each particle
     for particle in swarm:
          personal_best = particle.personal_best_position
          global_best = global_best_position
          informant_best = min(particle.informants, key = lambda p: p.personal_best_fitness).personal_best_position

          for i in range(dimensions):
               b = random.uniform(0, beta)
               c = random.uniform(0, gamma)
               d = random.uniform(0, delta)
               # Updating Velocity
               particle.velocity[i] = (alpha * particle.velocity[i] + 
                                       b* (personal_best[i] - particle.position[i]) +
                                       c * (informant_best[i] - particle.position[i]) + 
                                       d * (global_best[i] - particle.position[i]))

               # Updating position
               particle.position[i] += epsilon * particle.velocity[i] 

     if global_best_fitness < 0.95:
          break
     
# Decoding the optimized weights and biases
def weights_biases(global_best_position, layer_sizes):
    particle = Particle()
    particle.position = global_best_position
    weights, biases = particle.decode_position(layer_sizes)

    print('Optimized ANN weights and Biases: ')
    for i in range(len(weights)):
        print(f"Layer {i + 1} -> Layer{i + 2}: ")
        print(f"Weights:\n{weights[i]}")
        print(f"Biases:\n{biases[i]}")
        print("-" * 30)

weights_biases(global_best_position, layer_sizes)
print ('Global Best Fitness: ', global_best_fitness)
print('Mean Squared Error: ', fitness)
