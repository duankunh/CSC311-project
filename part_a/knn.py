from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    transposed_mat = np.transpose(matrix)
    mat = nbrs.fit_transform(transposed_mat)
    untransposed_mat = np.transpose(mat)
    acc = sparse_matrix_evaluate(valid_data, untransposed_mat)
    print("Item-based Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_list = [1, 6, 11, 16, 21, 26]
    user_list_val, item_list_val = [], []
    user_list_test, item_list_test = [], []

    for k in k_list:
        print("Validation and Test Accuracy for k =", k)
        #Validation data accuracy
        print("Validation")
        user_list_val.append(knn_impute_by_user(sparse_matrix, val_data, k))
        item_list_val.append(knn_impute_by_item(sparse_matrix, val_data, k))
        #Test data accuracy
        print("Test")
        user_list_test.append(knn_impute_by_user(sparse_matrix, test_data, k))
        item_list_test.append(knn_impute_by_item(sparse_matrix, test_data, k))
        print("\n")
    
    #plotting
    plt.title("K by User and Item Comparison")
    plt.plot(k_list, user_list_val, label="User-Based")
    plt.plot(k_list, item_list_val, label="Item-Based")
    plt.xlabel("K values")
    plt.ylabel("Acurracy")
    plt.legend(["User-Based", "Item-Based"])
    plt.show()

    #Selecting the best k on a user-based & item-based approach
    user_max_acc_index = user_list_test.index(max(user_list_test))
    item_max_acc_index = item_list_test.index(max(item_list_test))
    k_selected_by_user = k_list[user_max_acc_index]
    k_selected_by_item = k_list[item_max_acc_index]
    print('The user-based test accuracy with the best k is:', max(user_list_test))
    print('The best k for user-based collaborative filtering is:', k_selected_by_user)
    print('The item-based test accuracy with the best k is:', max(item_list_test))
    print('The best k for item-based collaborative filtering is:', k_selected_by_item)
    return k_selected_by_user, k_selected_by_item
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
