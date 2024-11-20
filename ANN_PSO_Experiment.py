import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time  # For tracking training time


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Initialize ANN weights and biases
def initialize_ANN(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
        biases.append(np.random.rand(layer_sizes[i + 1]) * 0.1)
    return weights, biases

# Feed forward propagation in ANN
def feed_forward(X, weights, biases, activation_functions):
    for i in range(len(weights)):
        X = activation_functions[i](np.dot(X, weights[i]) + biases[i])
    return X

# Decode PSO position into ANN weights and biases
def decode_position(position, layer_sizes):
    weights, biases, idx = [], [], 0
    for i in range(len(layer_sizes) - 1):
        weight_size = layer_sizes[i] * layer_sizes[i + 1]
        bias_size = layer_sizes[i + 1]
        weights.append(np.array(position[idx:idx + weight_size]).reshape(layer_sizes[i], layer_sizes[i + 1]))
        idx += weight_size
        biases.append(np.array(position[idx:idx + bias_size]))
        idx += bias_size
    return weights, biases

# Calculate MAE for regression evaluation
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# PSO particle class
class Particle:
    def __init__(self, dimensions):
        self.position = [random.uniform(-1, 1) for _ in range(dimensions)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)]
        self.personal_best_position = self.position[:]
        self.personal_best_fitness = float('inf')
        self.informants = []

# PSO optimization
def PSO(layer_sizes, activation_functions, X_train, y_train, swarm_size=20, iterations=100, alpha=0.5, beta=1.5, gamma=1.5, delta=0.5, epsilon=1):
    dimensions = sum(layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1)) + sum(layer_sizes[1:])
    swarm = [Particle(dimensions) for _ in range(swarm_size)]
    for particle in swarm:
        particle.informants = random.sample(swarm, k=min(5, swarm_size))

    global_best_position = None
    global_best_fitness = float('inf')

    for iteration in range(iterations):
        for particle in swarm:
            weights, biases = decode_position(particle.position, layer_sizes)
            predictions = feed_forward(X_train, weights, biases, activation_functions)
            fitness = mean_absolute_error(y_train, predictions)

            if fitness < particle.personal_best_fitness:
                particle.personal_best_position = particle.position[:]
                particle.personal_best_fitness = fitness
            if fitness < global_best_fitness:
                global_best_position = particle.position[:]
                global_best_fitness = fitness

        for particle in swarm:
            personal_best = np.array(particle.personal_best_position)
            global_best = np.array(global_best_position)
            position = np.array(particle.position)
            velocity = np.array(particle.velocity)

            b = random.uniform(0, beta)
            c = random.uniform(0, gamma)
            d = random.uniform(0, delta)
            particle.velocity = (
                alpha * velocity +
                b * (personal_best - position) +
                c * (random.choice(particle.informants).personal_best_position - position) +
                d * (global_best - position)
            )
            particle.position = position + epsilon * particle.velocity

    return global_best_position, global_best_fitness

# Split dataset into train and test
def custom_train_test_split(data, split_ratio=0.7):
    np.random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

# k-Fold Cross Validation
def k_fold_split(data, k=5):
    np.random.shuffle(data)
    fold_size = len(data) // k
    folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    return folds

def cross_validation_experiment(data, k=5, ann_configs=None, pso_configs=None):
    folds = k_fold_split(data, k)
    results = []

    for ann_config in ann_configs:
        for pso_config in pso_configs:
            fold_results = []

            for i in range(k):
                test_fold = folds[i]
                train_folds = np.concatenate([folds[j] for j in range(k) if j != i], axis=0)

                X_train, y_train = train_folds[:, :-1], train_folds[:, -1]
                X_test, y_test = test_fold[:, :-1], test_fold[:, -1]

                start_time = time.time()
                best_position, best_fitness = PSO(
                    ann_config["layers"],
                    ann_config["activation"],
                    X_train,
                    y_train,
                    swarm_size=pso_config["swarm_size"],
                    iterations=pso_config["iterations"]
                )
                end_time = time.time()

                weights, biases = decode_position(best_position, ann_config["layers"])
                predictions = feed_forward(X_test, weights, biases, ann_config["activation"])
                mae = mean_absolute_error(y_test, predictions)
                training_time = end_time - start_time

                fold_results.append({"MAE": mae, "Training Time": training_time})

            avg_mae = np.mean([res["MAE"] for res in fold_results])
            avg_time = np.mean([res["Training Time"] for res in fold_results])
            results.append({
                "ANN Config": ann_config["layers"],
                "PSO Config": f"Swarm: {pso_config['swarm_size']}, Iter: {pso_config['iterations']}",
                "Average MAE": avg_mae,
                "Average Training Time (s)": avg_time
            })

    for res in results:
        print(res)

# Load the dataset
def load_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    data = pd.read_excel(url).values
    return data

# Main function
def main():
    data = load_dataset()
    ann_configs = [
        {"layers": [8, 10, 1], "activation": [sigmoid, relu]},
        {"layers": [8, 20, 10, 1], "activation": [sigmoid, tanh, relu]},
    ]
    pso_configs = [
        {"swarm_size": 10, "iterations": 50},
        {"swarm_size": 20, "iterations": 100},
        {"swarm_size": 50, "iterations": 200},
    ]
    cross_validation_experiment(data, k=5, ann_configs=ann_configs, pso_configs=pso_configs)

if __name__ == "__main__":
    main()

