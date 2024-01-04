from time import perf_counter
from functools import reduce
from mnist import MNIST
import numpy as np


def build_fcnn(
    layers_sizes,
    activation,
    activation_derivative,
    X_train,
    y_train,
    batch_size,
    epochs_count=250,
    on_epoch_end=lambda epoch_number, learning_time, model: None,
    learning_rate=0.1,
):
    weights = [
        np.random.randn(next_layer_size, layer_size)
        for layer_size, next_layer_size in zip(layers_sizes[:-1], layers_sizes[1:])
    ]
    biases = [np.random.randn(layer_size, 1) for layer_size in layers_sizes[1:]]

    forward_pass_batch_activations = None

    def feedforward(batch):
        nonlocal weights, biases, forward_pass_batch_activations

        forward_pass_batch_activations = []
        activations = batch.T

        for layer_weights, layer_biases in zip(weights, biases):
            activations = activation(
                np.matmul(layer_weights, activations) + layer_biases
            )
            forward_pass_batch_activations.append(activations)

        return activations

    def backprop(X, outputs, y):
        nonlocal weights, forward_pass_batch_activations

        delta_l = outputs - y.T * activation_derivative(outputs)
        delta_ls = [delta_l]

        for layer_idx in range(len(layers_sizes) - 3, -1, -1):
            delta_l = np.matmul(
                weights[layer_idx + 1].T, delta_l
            ) * activation_derivative(forward_pass_batch_activations[layer_idx])
            delta_ls.insert(0, delta_l)

        for index, deltas_activations in enumerate(
            zip(delta_ls, [X.T, *forward_pass_batch_activations[:-1]])
        ):
            layer_delta_ls, previous_layer_batch_activations = deltas_activations

            weights[index] -= learning_rate * (
                np.matmul(
                    layer_delta_ls,
                    previous_layer_batch_activations.T,
                )
                / batch_size
            )
            biases[index] -= learning_rate * (
                np.sum(layer_delta_ls, axis=1, keepdims=True) / batch_size
            )

    def model(X):
        return feedforward(X).T

    def gradient_descent():
        for epoch_number in range(epochs_count):
            batches_indices = np.arange(len(X_train))
            np.random.shuffle(batches_indices)

            start_time = perf_counter()

            for batch_indices in [
                batches_indices[batch_idx : batch_idx + batch_size]
                for batch_idx in range(0, len(batches_indices), batch_size)
            ]:
                X = X_train[batch_indices]
                y = y_train[batch_indices]
                outputs = feedforward(X)
                backprop(X, outputs, y)

            on_epoch_end(epoch_number, perf_counter() - start_time, model)

    gradient_descent()

    return model


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    sigmoid_given_z = sigmoid(z)
    return sigmoid_given_z * (1 - sigmoid_given_z)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def fetch_mnist():
    mndata = MNIST("data")
    mndata.gz = True
    mndata.load_training()
    mndata.load_testing()

    X_train = np.array(mndata.train_images)
    y_train = np.array(mndata.train_labels)
    X_test = np.array(mndata.test_images)
    y_test = np.array(mndata.test_labels)

    return X_train, y_train, X_test, y_test


def main():
    print("Fetching MNIST data...")
    X_train, y_train, X_test, y_test = fetch_mnist()

    def evaluate_accuracy(model):
        model_outputs = model(X_test)

        return reduce(
            lambda correct_count, outputs: correct_count + (np.argmax(outputs[0]) == np.argmax(outputs[1])),
            zip(model_outputs, y_test),
            0,
        ) / len(X_test)

    pixels_count = X_train.shape[1]
    model = build_fcnn(
        layers_sizes=[
            pixels_count,
            100,
            50,
            10,
        ],
        activation=tanh,
        activation_derivative=tanh_derivative,
        X_train=X_train,
        y_train=y_train,
        batch_size=64,
        epochs_count=10,
        on_epoch_end=lambda epoch_number, learning_time, model: print(
            f"Epoch {epoch_number} finished in {learning_time} with test accuracy: {evaluate_accuracy(model)}"
        ),
    )

    print(f"Final test accuracy: {evaluate_accuracy(model)}")


if __name__ == "__main__":
    main()
