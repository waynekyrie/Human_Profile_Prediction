import numpy as np
import torch
import matplotlib.pyplot as plt

def create_inout_sequences(input_data, seq_len, input_size, output_step):
    inout_seq = []
    for data in input_data:
        data_seq = []
        L = len(data)
        for i in range(L-seq_len-output_step+1):
            train_seq = data[i:i+seq_len, :input_size]
            train_label = data[i+seq_len:i+seq_len+output_step, :input_size]
            data_seq.append((train_seq ,train_label))
        inout_seq.append(data_seq)
    return inout_seq

def load_data(data_set):
    data_all = []
    for data_fname in data_set:
        human_id = data_fname[0]
        task_id = data_fname[1]
        trial = data_fname[2]

        data = np.loadtxt('../Data_Collection/data/subject'+str(human_id)+'_'+str(task_id)+'_'+str(trial)+'_instruct/data.txt')
        data = data[:, :6]
        data_col = data.shape[1] # should equal to output_size
        data_all.append(data)
    return data_all, data_col

def load_data_squeeze(data_set, data_col):
    data_all = np.zeros((1, data_col))
    for data_fname in data_set:
        human_id = data_fname[0]
        task_id = data_fname[1]
        trial = data_fname[2]

        data = np.loadtxt('../Data_Collection/data/subject'+str(human_id)+'_'+str(task_id)+'_'+str(trial)+'_instruct/data.txt')
        data = data[:, :6]
        data_all = np.concatenate((data_all, data), axis=0)
    data_all = data_all[1:, :]
    return data_all

def fit_scaler():
    data_max = np.array([0.35, 0.2, 0.95, 0.05, 0.2, 0.95])
    data_min = np.array([-0.15, -0.2, 0.4, -0.3, -0.15, 0.4])
    return data_min, data_max

def scale_data(data_all, data_col, scaler):
    data_normalized_all = []
    for data in data_all:
        data_normalized = scaler.transform(data.reshape(-1, data_col))
        data_normalized = torch.FloatTensor(data_normalized).view(-1, data_col)
        data_normalized_all.append(data_normalized)
    return data_normalized_all

def inverse_scale_data(data, data_col, scaler):
    data_inverse = scaler.inverse_transform(np.array(data).reshape(-1, data_col))
    return data_inverse

def visualize_predictions(test_data, actual_predictions, predict_step, input_step):
    # Visualize Results
    fig, axs = plt.subplots(6) # visualize lx, rx, ly, ry, lz, rz
    fig.suptitle('Prediction Step: '+str(predict_step))
    test_data_size = len(test_data)

    # Plot X
    axs[0].set(ylabel='Left X Position')
    axs[0].grid(True)
    axs[0].autoscale(axis='x', tight=True)
    axs[0].plot(test_data[:, 0], label='Test Data')
    if(predict_step <= 1):
        axs[0].plot(actual_predictions[:, :, 0], label='Predictions')
    else:
        for i in range(input_step, test_data_size, predict_step+5):
            x = np.arange(i, i+predict_step, 1)
            if(i == input_step):
                axs[0].plot(x, actual_predictions[i, :, 0], label='Predictions')
            else:
                axs[0].plot(x, actual_predictions[i, :, 0])
    axs[0].legend(loc="upper right", shadow=True, fancybox=True)

    axs[1].set(ylabel='Right X Position')
    axs[1].grid(True)
    axs[1].autoscale(axis='x', tight=True)
    axs[1].plot(test_data[:, 3])
    if(predict_step <= 1):
        axs[1].plot(actual_predictions[:, :, 3])
    else:
        for i in range(input_step, test_data_size, predict_step+5):
            x = np.arange(i, i+predict_step, 1)
            if(i == input_step):
                axs[1].plot(x, actual_predictions[i, :, 3])
            else:
                axs[1].plot(x, actual_predictions[i, :, 3])

    # Plot Y
    axs[2].set(ylabel='Left Y Position')
    axs[2].grid(True)
    axs[2].autoscale(axis='x', tight=True)
    axs[2].plot(test_data[:, 1])
    if(predict_step <= 1):
        axs[2].plot(actual_predictions[:, :, 1])
    else:
        for i in range(input_step, test_data_size, predict_step+5):
            x = np.arange(i, i+predict_step, 1)
            axs[2].plot(x, actual_predictions[i, :, 1])

    axs[3].set(ylabel='Right Y Position')
    axs[3].grid(True)
    axs[3].autoscale(axis='x', tight=True)
    axs[3].plot(test_data[:, 4])
    if(predict_step <= 1):
        axs[3].plot(actual_predictions[:, :, 4])
    else:
        for i in range(input_step, test_data_size, predict_step+5):
            x = np.arange(i, i+predict_step, 1)
            axs[3].plot(x, actual_predictions[i, :, 4])

    # Plot Z
    axs[4].set(ylabel='Left Z Position')
    axs[4].grid(True)
    axs[4].autoscale(axis='x', tight=True)
    axs[4].plot(test_data[:, 2])
    if(predict_step <= 1):
        axs[4].plot(actual_predictions[:, :, 2])
    else:
        for i in range(input_step, test_data_size, predict_step+5):
            x = np.arange(i, i+predict_step, 1)
            axs[4].plot(x, actual_predictions[i, :, 2])

    axs[5].set(xlabel='Time', ylabel='Right Z Position')
    axs[5].grid(True)
    axs[5].autoscale(axis='x', tight=True)
    axs[5].plot(test_data[:, 5])
    if(predict_step <= 1):
        axs[5].plot(actual_predictions[:, :, 5])
    else:
        for i in range(input_step, test_data_size, predict_step+5):
            x = np.arange(i, i+predict_step, 1)
            axs[5].plot(x, actual_predictions[i, :, 5])

def show_visualizations():
    plt.show()