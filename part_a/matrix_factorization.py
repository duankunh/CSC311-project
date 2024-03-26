from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    prediction = np.dot(u[n], z[q])

    # Compute error
    error = prediction - c
    
    # Update parameters u and z using gradients
    u_gradient = -error * z[q]
    z_gradient = -error * u[n]
    u[n] -= lr * u_gradient
    z[q] -= lr * z_gradient

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, valid_data, plot=True):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if i % 100 == 0 and plot:
            mat = np.dot(u, np.transpose(z))
            # Record the current training and validation loss values.
            train_loss.append(squared_error_loss(train_data, u, z))
            valid_loss.append(squared_error_loss(valid_data, u, z))
            train_acc.append(sparse_matrix_evaluate(train_data, mat))
            valid_acc.append(sparse_matrix_evaluate(valid_data, mat))

    if plot:
        plt.title("Training Curve Showing Training and Validation Loss at each Iteration")
        plt.plot(train_loss, label="Training Loss")
        # plt.plot(valid_loss, label="Validation Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

        plt.title("Training Curve Showing Training and Validation Accuracy at each Iteration")
        plt.plot(train_acc, label="Training Accuracy")
        # plt.plot(valid_acc, label="Validation Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.show()
    mat = np.dot(u, np.transpose(z))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat

def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # ks = [5, 7, 9, 14, 16]
    # validation_accuracy = []
    # test_accuracy = []
    # print("Testing for k-values: ", ks)
    # for k in ks:
    #     reconstructed_matrix = svd_reconstruct(train_matrix, k)
    #     validation_accuracy.append(sparse_matrix_evaluate(val_data, reconstructed_matrix))
    #     test_accuracy.append(sparse_matrix_evaluate(test_data, reconstructed_matrix))
    # best_k_index = np.argmax(validation_accuracy)
    # print(f"Best k is k = {ks[best_k_index]}")
    # print(f"Validation accuracy: {validation_accuracy[best_k_index]}")
    # print(f"Test accuracy: {test_accuracy[best_k_index]}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    ##############################
    #######################################

    losses = []
    k = 10
    als_matrix = als(train_data, k, 0.01, 3000, val_data, True)
    print(f"Validation accuracy: {sparse_matrix_evaluate(val_data, als_matrix)}")
    # ks = [1, 5, 9, 13, 15]
    # validation_accuracy = []
    # for k in ks:
    #     als_matrix = als(train_data, k, 0.66, 10, [])
    #     validation_accuracy.append(sparse_matrix_evaluate(val_data, als_matrix))

    # best_k_index = np.argmax(validation_accuracy)
    # best_matrix = als(train_data, ks[best_k_index], 0.66, 10, losses)
    # print(f"Best k is k = {ks[best_k_index]}")
    # print(f"Validation accuracy: {validation_accuracy[best_k_index]}")
    # print(f"Test accuracy: {sparse_matrix_evaluate(test_data, als_matrix)}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
