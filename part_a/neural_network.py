from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        sigmoid = nn.Sigmoid()
        g_bias = torch.unsqueeze(self.g.bias, 1)  # make shape = 10, 1
        h_bias = torch.unsqueeze(self.h.bias, 1)  # make shape = 1774, 1
        g = sigmoid(self.g.weight @ inputs.mT + g_bias)
        out = sigmoid(self.h.weight @ g + h_bias)
        out = out.T
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    # NOTE: for plotting
    epochs = []
    train_cost = []
    valid_accuracy = []

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].numpy())
            nan_mask = torch.from_numpy(nan_mask).unsqueeze(0)  # Unsqueeze mask to match input shape

            target_masked = torch.where(nan_mask, output, target)  # Use torch.where to apply mask

            loss = torch.sum((output - target_masked) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        # TODO: add regularization
        train_loss += lamb / 2 * model.get_weight_norm()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        # NOTE: for plotting
        epochs.append(epoch)
        train_cost.append(train_loss)
        valid_accuracy.append(valid_acc)
    return epochs, train_cost, valid_accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # k in {10, 50, 100, 200, 500}
    k = 10
    # num questions in valid_data
    num_question = train_matrix.shape[1]
    # print(num_question)
    model = AutoEncoder(num_question, k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 125
    lamb = 0.01  # e) tune {0.001, 0.01, 0.1, 1}

    epochs, train_cost, valid_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    # getting test accuracy of the model
    acc = evaluate(model, zero_train_matrix, test_data)
    print('Test accuracy: ', acc)
    # plot
    fig, ax = plt.subplots(1, 2)
    # train loss vs epochs
    ax[0].plot(epochs, train_cost)
    ax[0].set_xlabel('Epoch number')
    ax[0].set_ylabel('Train cost')

    # valid acc vs epochs
    ax[1].plot(epochs, valid_acc)
    ax[1].set_xlabel('Epoch number')
    ax[1].set_ylabel('Valid accuracy')

    fig.tight_layout()

    plt.savefig('q3d.png')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
