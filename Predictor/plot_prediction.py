import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np

test_data = np.loadtxt('0test.txt')
predictions = np.loadtxt('0predict30.txt')
output_size = 6
output_step = 30

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(test_data)):
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.9)

    xdata = []
    ydata = []
    zdata = []
    x_predl = []
    y_predl = []
    z_predl = []
    x_predr = []
    y_predr = []
    z_predr = []
    x_truel = []
    y_truel = []
    z_truel = []
    x_truer = []
    y_truer = []
    z_truer = []


    lwrist = test_data[i, :3]
    rwrist = test_data[i, 3:]
    prediction = predictions[i, :]

    xdata.append(lwrist[0])
    xdata.append(rwrist[0])

    ydata.append(lwrist[1])
    ydata.append(rwrist[1])

    zdata.append(lwrist[2])
    zdata.append(rwrist[2])

    for j in range(output_step):
        pred_j = prediction[j*output_size:(j+1)*output_size]
        x_predl.append(pred_j[0])
        x_predr.append(pred_j[3])
        y_predl.append(pred_j[1])
        y_predr.append(pred_j[4])
        z_predl.append(pred_j[2])
        z_predr.append(pred_j[5])

        try:
            x_truel.append(test_data[i+j, 0])
            x_truer.append(test_data[i+j, 3])
            y_truel.append(test_data[i+j, 1])
            y_truer.append(test_data[i+j, 4])
            z_truel.append(test_data[i+j, 2])
            z_truer.append(test_data[i+j, 5])
        except Exception as e:
            print(e)

    ax.scatter(xdata, ydata, zdata, c = 'black', s=200)
    ax.plot(x_predl, y_predl, z_predl, c = 'red')
    ax.plot(x_predr, y_predr, z_predr, c = 'blue')
    ax.plot(x_truel, y_truel, z_truel, c='green')
    ax.plot(x_truer, y_truer, z_truer, c='green')
    plt.draw()
    plt.pause(0.01)
    
    ax.cla()
    