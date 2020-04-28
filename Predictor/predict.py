import torch
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler
import utils as util
import RNN_LSTM as LSTM

train_data_set = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1],
                  [0, 3, 0], [0, 3, 1], [0, 4, 0], [0, 4, 1]]

test_data_set = [[0, 0, 0], [1, 0, 0]]#[0, 1, 1], [1, 0, 0], [1, 1, 0]]

scale_data_set = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 2, 1],
                  [0, 3, 0], [0, 3, 1], [0, 4, 0], [0, 4, 1]]

sequence_length = 30
input_size = 6 # dimension of input
hidden_size = 128
num_layers = 1
output_step = 1 # predict step
output_size = 6 # dimension of output
num_epochs = 150
learning_rate = 0.001

save_model = True # If true, save trained model every 20 epoch
load_saved_model = True # If true, load model and skip training
model_path = 'pred1/model199.tar'##'pred20/model121.tar'#'pred10/model141.tar'#'pred5/model181.tar'#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_data, train_data_col = util.load_data(train_data_set)
scale_data_squeezed = util.load_data_squeeze(scale_data_set, train_data_col)

test_data, test_data_col = util.load_data(test_data_set)

# Scale data
scaler = MinMaxScaler(feature_range=(-5, 5))
scaler.fit(scale_data_squeezed)
#scaler.data_min_, scaler.data_max_ = util.fit_scaler()
train_data_normalized = util.scale_data(train_data, train_data_col, scaler)
train_inout_seq = util.create_inout_sequences(train_data_normalized, sequence_length, input_size, output_step)
test_data_normalized = util.scale_data(test_data, test_data_col, scaler)

if(not load_saved_model):
    model = LSTM.RNN(input_size, sequence_length, hidden_size, num_layers, output_step, output_size, device).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    print('Start training!')
    time_start = time.time()
    for i in range(num_epochs):
        for inout_seq in train_inout_seq:
            for seq, labels in inout_seq:
                seq = seq.view(1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                y_pred = model(seq)

                loss = loss_function(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if(i%20 == 1):
            print(f'epoch: {i:1} loss: {loss.item():10.10f}', f'   Cost time: {time.time()-time_start:1.3f}s', )
            if(save_model):
                torch.save({'epoch': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, './output_models/model'+str(i)+'.tar')

    print(f'epoch: {i:1} loss: {loss.item():10.10f}')
    if(save_model):
        torch.save({'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, './output_models/model'+str(i)+'.tar')
    print(f'Training time: {time.time()-time_start:1.3f}')
else:
    if(model_path[-1] == 'h'):
        model = LSTM.RNN(input_size, sequence_length, hidden_size, num_layers, output_step, output_size, device).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint.state_dict())
    else:
        model = LSTM.RNN(input_size, sequence_length, hidden_size, num_layers, output_step, output_size, device).to(device)
        loss_function = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

# Evaluate
print('Start Evaluating!')
model.eval()

# Evaluate for each traj profile
for test_idx in range(len(test_data_set)):
    test_inputs = test_data_normalized[test_idx][:sequence_length, :test_data_col].tolist()
    test_data_size = len(test_data[test_idx])
    test_output = []
    for i in range(test_data_size):
        test_inputs.append(test_data_normalized[test_idx][i, :test_data_col].tolist())
        seq = torch.FloatTensor(test_inputs[-sequence_length:]).view(1, sequence_length, input_size).to(device)
        with torch.no_grad():
            predict = model(seq)
            cur_predict = []
            for l in range(output_step):
                step_predict = []
                for j in range(output_size):
                    step_predict.append(predict[l, j].item())
                cur_predict.append(step_predict)
            test_output.append(cur_predict)      
    actual_predictions = util.inverse_scale_data(test_output, test_data_col, scaler)
    actual_predictions = actual_predictions.reshape(test_data_size, output_step, output_size)

    # Visualize Results
    util.visualize_predictions(test_data[test_idx], actual_predictions, output_step, sequence_length)
util.show_visualizations()