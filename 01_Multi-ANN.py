import numpy as np

#sigmoid activation function and  it's derivation
def sigmoid(x):
    return 1/(1+ np.exp(-x))

#relu activation function and it's derivation
def relu(x): 
    return np.maximum(0,x)

#tanh activation function and it's derivation
def tanh(x):  
      return np.tanh(x)

def __init__(layer_sizes):   
        weights = []         #initializes an empty list where each item will store the weight
        biases = []          #initializes an empty list where each item will store the biases
        
        for i in range(len(layer_sizes) -1):          #This loop excludes the input layer as weights & biases connect layers to one another
            weights.append(np.random.rand(layer_sizes[i], layer_sizes[i+1]) *0.1)       #initializes weight between layer i & layer i+1 using random values & multiplying by 0.1 makes the initial bias value down
            biases.append(np.random.rand(layer_sizes[i+1])* 0.1)                        #initializes the bias in layer i+1 with random values
        return weights, biases

def feed_forward(X, weights, biases, activation_functions):
        for i in range(len(weights)):
             a = np.dot(X, weights[i]) + biases[i]  #calculates the input data X with weight of the currentlayer and ten adds the bias
             X = activation_functions[i](a)  #Applies activation function for the current layer 
        return X
    
def neural_network(layer_sizes, activation_functions, X):
      #Initializing weights and biases
      weights, biases= __init__(layer_sizes)

      #feed_forward propogation
      output = feed_forward(X, weights, biases, activation_functions)

      return output


##Defining the architecture
layer_sizes = [5, 3, 3, 1] #5 input features, 3 nodes in 1st hidden layers, 3 nodes in 2nd hidden layer, 1 output
activation_functions = [sigmoid, tanh, relu]   #Activation Functions for each layer sigmoid for 1st hidden layer, tanh for 2nd hidden layer and relu for output layer

#Generating random input data
input_data = np.random.randn(3,5)  # 3 examples, 5 input features each


#Output
output = neural_network(layer_sizes, activation_functions, input_data)
print("Layer Sizes : ", layer_sizes)
print("Activation functions: ", activation_functions)
print("Inputs : ", input_data)
print("Output : ", output)


