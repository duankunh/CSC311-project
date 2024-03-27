from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_likelihood = 0.
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
        # Calculate the log likelihood term for a single student-question pair
        p = is_correct * (theta[user_id] - beta[question_id]) - np.log(
            1 + np.exp(theta[user_id] - beta[question_id]))
        # Update the log likelihood sum
        log_likelihood += p

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_likelihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    # Initialize gradients for theta and beta
    g_theta = np.zeros_like(theta)
    g_beta = np.zeros_like(beta)
    # Compute gradients

    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
        z = theta[user_id] - beta[question_id]
        p = sigmoid(z)
        g_theta[user_id] += is_correct - p
        g_beta[question_id] += p - is_correct
    theta += lr * g_theta
    beta -= lr * g_beta
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, train_log_likelihoods,
    val_log_likelihoods)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(max(data['user_id']) + 1)
    beta = np.zeros(max(data['question_id']) + 1)

    val_acc_lst = []

    # answer for (c)
    train_log_likelihoods = []
    val_log_likelihoods = []

    for i in range(iterations):

        # answer for (c)
        train_neg_lld = neg_log_likelihood(data, theta, beta)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        train_log_likelihoods.append(-train_neg_lld)
        val_log_likelihoods.append(-val_neg_lld)
        # the original code part
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_log_likelihoods, val_log_likelihoods


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def plot_log_likelihoods(train_log_likelihoods, val_log_likelihoods):
    plt.figure(figsize=(10, 5))
    plt.plot(train_log_likelihoods, label='Training Log-likelihood')
    plt.plot(val_log_likelihoods, label='Validation Log-likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title('Training and Validation Log-likelihoods')
    plt.legend()
    plt.show()


def capable_difficulty_curve(theta, beta, question_ids):
    """Function for (d) """
    theta_vals = np.linspace(min(theta), max(theta), 100)
    plt.figure(figsize=(10, 6))
    for q_id in question_ids:
        p_correct = sigmoid(theta_vals - beta[q_id])
        plt.plot(theta_vals, p_correct,
                 label=f"Question {q_id + 1} (Difficulty: {beta[q_id]:.2f})")
    plt.title("Item Characteristic Curves")
    plt.xlabel("Student Ability (Theta)")
    plt.ylabel("Probability of Correct Response")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # lr = 0.0009
    lr = 0.002
    niter = 50
    theta, beta, val_acc_lst, train_log_likelihoods, val_log_likelihoods = \
        irt(train_data, val_data, lr, niter)
    # for (c)
    plot_log_likelihoods(train_log_likelihoods, val_log_likelihoods)

    val_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)
    print(f"validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
    print(max(beta))
    print(min(beta))
    max_index = np.argmax(beta)
    min_index = np.argmin(beta)
    print(f"Index of maximum value in beta: {max_index}")
    print(f"Index of minimum value in beta: {min_index}")
    print(beta[max_index])
    print(beta[min_index])
    print(beta[1444])
    print(beta[1410])
    print(beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # for (d)
    selected_question_ids = [1410, 2,
                             1444]
    capable_difficulty_curve(theta, beta, selected_question_ids)

    # The shapes of the curves in the graph are typical of Item Characteristic
    # Curves (ICCs) from Item Response Theory. The blue curve represents an
    # relatively easy question, with a very high probability of correct
    # responses across all ability levels, indicating that it even students with
    # low abilities can perform better than the rest of two questions. The
    # orange curve is more typical, representing an average-difficulty
    # question(since it is very closed to zero), where students of higher
    # ability have a higher chance of answering correctly.
    # The orange curve suggests an very difficult question, requiring a
    # high ability level to have a low chance to solve the problem compared
    # with above two. So overall, the three curves represents when students'
    # ability is fixed, how many chance they have to solve the questions with
    # varies difficulty, which is good reference to check if the difficulty of
    # test or final exam is reasonable or not
    # Implement part (d)                                                #
    #####################################################################
    # pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
