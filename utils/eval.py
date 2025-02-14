import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr


# Evaluation functions as revised earlier
def get_eval(test_pred, ground_truth):
    mse = mean_squared_error(ground_truth, test_pred)
    pcc_list = []
    scc_list = []
    pcc, _ = pearsonr(ground_truth, test_pred)
    scc, _ = spearmanr(ground_truth, test_pred)
    return [mse, pcc, scc]


def get_eval_deletion(test_pred, ground_truth):
    test_pred = test_pred[:, :535]
    ground_truth = ground_truth[:, :535]
    # TODO 535 correct?
    mse = mean_squared_error(ground_truth, test_pred)
    pcc_list = []
    scc_list = []
    for i in range(ground_truth.shape[1]):
        pcc, _ = pearsonr(ground_truth[:, i], test_pred[:, i])
        scc, _ = spearmanr(ground_truth[:, i], test_pred[:, i])
        pcc_list.append(pcc)
        scc_list.append(scc)
    pcc_mean = np.mean(pcc_list)
    scc_mean = np.mean(scc_list)
    return [mse, pcc_mean, scc_mean]

def get_eval_1bpindels(test_pred, ground_truth):
    mse = mean_squared_error(ground_truth, test_pred)
    pcc_list = []
    scc_list = []
    for i in range(ground_truth.shape[1]):
        pcc, _ = pearsonr(ground_truth[i, i], test_pred[:, i])
        scc, _ = spearmanr(ground_truth[:, i], test_pred[:, i])
        pcc_list.append(pcc)
        scc_list.append(scc)
    pcc_mean = np.mean(pcc_list)
    scc_mean = np.mean(scc_list)
    return [mse, pcc_mean, scc_mean]