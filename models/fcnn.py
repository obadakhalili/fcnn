import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from time import perf_counter
from functools import reduce


def build_fcnn(
    layers_sizes,
    activation,
    activation_derivative,
    data,
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

    forward_pass_batch_activations = []

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


    def gradient_descent():
        for epoch_number in range(epochs_count):
            np.random.shuffle(data)
            start_time = perf_counter()

            for batch in [
                data[batch_idx : batch_idx + batch_size]
                for batch_idx in range(0, len(data), batch_size)
            ]:
                X = np.array([sample[0] for sample in batch])
                y = np.array([sample[1] for sample in batch])

                outputs = feedforward(X)
                backprop(X, outputs, y)

            on_epoch_end(epoch_number, perf_counter() - start_time, feedforward)

    gradient_descent()

    return feedforward


# must accept an primitive number or a numpy array of numbers
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    sigmoid_given_z = sigmoid(z)
    return sigmoid_given_z * (1 - sigmoid_given_z)

def main():
    data = datasets.load_digits()

    # NOTE: how does this work?
    data.target = np.eye(10)[data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        data.images.reshape((data.images.shape[0], -1)),
        data.target,
        test_size=0.3,
        shuffle=False,
    )

    pixels_count = data.images.shape[1] * data.images.shape[2]

    def evaluate_accuracy(model):
        model_outputs = model(X_test).T

        return reduce(
            lambda correct_count, outputs: correct_count + (np.argmax(outputs[0]) == np.argmax(outputs[1])),
            zip(model_outputs, y_test),
            0,
        ) / len(X_test)

    model = build_fcnn(
        layers_sizes=[
            pixels_count,
            100,
            50,
            10,
        ],
        activation=sigmoid,
        activation_derivative=sigmoid_derivative,
        data=list(zip(X_train, y_train)),
        batch_size=8,
        on_epoch_end=lambda epoch_number, learning_time, model: print(
            f"Epoch {epoch_number} finished in {learning_time} with accuracy: {evaluate_accuracy(model)}"
        ),
    )

    print(f"Final accuracy: {evaluate_accuracy(model)}")


if __name__ == "__main__":
    main()
