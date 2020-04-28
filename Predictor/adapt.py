import torch
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler
import utils as util
import LSTM_adapt as LSTM
import numpy as np


scale_data_set = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1],
                  [0, 3, 0], [0, 3, 1], [0, 4, 0], [0, 4, 1]]

test_data_set = [[1, 0, 0]]

lamda = 1 # Forgetting factor

sequence_length = 30
input_size = 6 # dimension of input
hidden_size = 128
num_layers = 1
output_step = 30 # predict step
output_size = 6 # dimension of output
learning_rate = 0.001

model_path = 'pred30/model121.tar'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
scale_data_squeezed = util.load_data_squeeze(scale_data_set, input_size)
test_data, test_data_col = util.load_data(test_data_set)

# Construct scaler
scaler = MinMaxScaler(feature_range=(-5, 5))
scaler.fit(scale_data_squeezed)
test_data_normalized = util.scale_data(test_data, test_data_col, scaler)

# Load model
if(model_path[-1] == 'r'):
    model = LSTM.RNN(input_size, sequence_length, hidden_size, num_layers, output_step, output_size, device).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model = LSTM.RNN(input_size, sequence_length, hidden_size, num_layers, output_step, output_size, device).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint.state_dict())


model.eval()
print('Start adapting!')
param = model.state_dict()['linear.weight']
bias = model.state_dict()['linear.bias'].cpu()

F_pre = 1000 * np.identity(hidden_size)
x_pre = np.zeros((hidden_size, 1)) # 128x1

# Evaluate for each traj profile
for test_idx in range(len(test_data_set)):
    test_inputs = test_data_normalized[test_idx][:sequence_length, :test_data_col].tolist()
    ground_truth = test_data_normalized[test_idx][:, :test_data_col].tolist()
    test_data_size = len(test_data[test_idx])
    test_output = []
    for i in range(test_data_size):
        test_inputs.append(test_data_normalized[test_idx][i, :test_data_col].tolist())
        seq = torch.FloatTensor(test_inputs[-sequence_length:]).view(1, sequence_length, input_size).to(device)
        with torch.no_grad():
            model.x_pre = x_pre

            # new
            param_pre = model.state_dict()['linear.weight'][:, :].cpu()
            upper = np.matmul(np.matmul(np.matmul(F_pre, x_pre), x_pre.T), F_pre)
            bottom = np.matmul(np.matmul(x_pre.T, F_pre), x_pre)
            Fk = 1/lamda*(F_pre-upper/(lamda+bottom))

            model.x_pre = x_pre
            predict, xt = model(seq)

            adapt = 1
            try:
                cur_measurement = np.asarray(ground_truth)[i-output_step+1:i+1, :test_data_col]
                cur_measurement = np.flip(cur_measurement, 0)
                cur_predict = np.asarray(predict.cpu())#np.asarray(test_output[-1])
                
                et = cur_measurement - cur_predict
                et = np.reshape(et, (1, output_step*output_size))
                noise = np.random.normal(size=(output_step, output_size))*0.1
                '''noise = util.scale_data(noise, output_size, scaler)
                for idx in range(len(noise)):

                    noise[idx] = np.asarray(noise[idx])
                noise = np.asarray(noise) '''
                noise = noise.reshape((1, output_size*output_step))
                et[6:] += noise[6:]

            except Exception as e:
                adapt = 0
                print(e)
                print("prediction step not enough!")
                et = np.zeros((1, output_step*output_size))
            if(adapt):
                new_param = param_pre + np.matmul(np.matmul(Fk, x_pre), et).T
                model.state_dict()['linear.weight'][:, :] = new_param[:, :]
                F_pre = Fk
            x_pre = np.asarray(xt)
            
            cur_predict = []
            for l in range(output_step):
                step_predict = []
                for j in range(output_size):
                    step_predict.append(predict[l, j].item())
                cur_predict.append(step_predict)
            test_output.append(cur_predict)
    actual_predictions = util.inverse_scale_data(test_output, test_data_col, scaler)
    actual_predictions = actual_predictions.reshape(test_data_size, output_step, output_size)
    
    error_l, error_r = util.calculate_rmse(test_data[test_idx], actual_predictions)
    print(error_l, error_r)
    # Visualize Results
    util.visualize_predictions(test_data[test_idx][:test_data_size], actual_predictions, output_step, sequence_length)

util.show_visualizations()