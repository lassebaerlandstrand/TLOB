import numpy as np
import constants as cst
import torch

  
def fi_2010_load(path, seq_size, horizon, all_features):
    dec_data = np.loadtxt(path + "/Train_Dst_NoAuction_ZScore_CF_7.txt")
    full_train = dec_data[:, :int(dec_data.shape[1] * cst.SPLIT_RATES[0])]
    full_val = dec_data[:, int(dec_data.shape[1] * cst.SPLIT_RATES[0]):]
    dec_test1 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_7.txt')
    dec_test2 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_8.txt')
    dec_test3 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_9.txt')
    full_test = np.hstack((dec_test1, dec_test2, dec_test3))
    
    if horizon == 10:
        tmp = 5
    elif horizon == 20:
        tmp = 4
    elif horizon == 30:
        tmp = 3
    elif horizon == 50:
        tmp = 2
    elif horizon == 100:
        tmp = 1
    else:
        raise ValueError("Horizon not found")
    
    train_labels = full_train[-tmp, :].flatten()
    val_labels = full_val[-tmp, :].flatten()
    test_labels = full_test[-tmp, :].flatten()
    
    train_labels = train_labels[seq_size-1:] - 1
    val_labels = val_labels[seq_size-1:] - 1
    test_labels = test_labels[seq_size-1:] - 1
    if all_features:
        train_input = full_train[:144, :].T
        val_input = full_val[:144, :].T
        test_input = full_test[:144, :].T
    else:
        train_input = full_train[:40, :].T
        val_input = full_val[:40, :].T
        test_input = full_test[:40, :].T
    train_input = torch.from_numpy(train_input).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_input = torch.from_numpy(val_input).float()
    val_labels = torch.from_numpy(val_labels).long()
    test_input = torch.from_numpy(test_input).float()
    test_labels = torch.from_numpy(test_labels).long()
    return train_input, train_labels, val_input, val_labels, test_input, test_labels


def fi_2010_load_multi(path, seq_size, all_features):
    """Load FI-2010 data returning all 4 horizon labels stacked.

    In FI-2010 the label rows are (from the bottom of the feature matrix):
      row -5 = h10, -4 = h20, -3 = h30, -2 = h50, -1 = h100.
    We use rows [-5, -4, -2, -1] -> [h10, h20, h50, h100].

    Returns:
        train_input, train_labels(N,4), val_input, val_labels(N,4),
        test_input, test_labels(N,4)
    """
    dec_data = np.loadtxt(path + "/Train_Dst_NoAuction_ZScore_CF_7.txt")
    full_train = dec_data[:, :int(dec_data.shape[1] * cst.SPLIT_RATES[0])]
    full_val = dec_data[:, int(dec_data.shape[1] * cst.SPLIT_RATES[0]):]
    dec_test1 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_7.txt')
    dec_test2 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_8.txt')
    dec_test3 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_9.txt')
    full_test = np.hstack((dec_test1, dec_test2, dec_test3))

    # Horizon row indices (negative): h10=-5, h20=-4, h50=-2, h100=-1
    horizon_rows = [-5, -4, -2, -1]

    def extract_labels(data):
        labels = np.stack([data[r, :].flatten() for r in horizon_rows], axis=1)  # (T, 4)
        labels = labels[seq_size-1:] - 1  # align with inputs, convert 1-indexed to 0-indexed
        return labels

    train_labels = extract_labels(full_train)
    val_labels = extract_labels(full_val)
    test_labels = extract_labels(full_test)

    if all_features:
        train_input = full_train[:144, :].T
        val_input = full_val[:144, :].T
        test_input = full_test[:144, :].T
    else:
        train_input = full_train[:40, :].T
        val_input = full_val[:40, :].T
        test_input = full_test[:40, :].T

    train_input = torch.from_numpy(train_input).float()
    val_input = torch.from_numpy(val_input).float()
    test_input = torch.from_numpy(test_input).float()
    train_labels = torch.from_numpy(train_labels.astype(np.int64)).long()
    val_labels = torch.from_numpy(val_labels.astype(np.int64)).long()
    test_labels = torch.from_numpy(test_labels.astype(np.int64)).long()
    return train_input, train_labels, val_input, val_labels, test_input, test_labels
