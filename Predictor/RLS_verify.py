import numpy as np
import matplotlib.pyplot as plt

x_range = 100
true_param = np.matrix([[1, 2, 3],
                        [4, 5, 6]])

param = np.ones((2, 3))

y_est_list = []
y_true_list = []

F_pre = 1000 * np.identity(2)
x_pre = np.ones((2, 1))
x_pre_true = x_pre
et = np.zeros((1, 3))
lamda = 0.1

for i in range(2, x_range):
    y_est = np.matmul(x_pre.T, param)
    y_true = np.matmul(x_pre_true.T, true_param)
    
    y_true_list.append(y_true)
    y_est_list.append(y_est)
    et = y_true-y_est

    true_param=true_param+0.01

    phi_t = x_pre.T
    phi = phi_t.T
    upper = np.matmul(np.matmul(np.matmul(F_pre, phi), phi_t), F_pre)
    bottom = np.matmul(np.matmul(phi_t, F_pre), phi) + lamda
    Ft = (F_pre - upper/bottom)/lamda
    d_param = np.matmul(Ft, phi)*et

    param = param + d_param
    n = np.random.normal(size=2)
    n = np.reshape(n, (2, 1))
    x_cur = np.matrix([[i], [i]])
    x_cur_n = x_cur+n

    F_pre = Ft
    x_pre = x_cur_n
    x_pre_true = x_cur

y_est_list = np.asarray(y_est_list)
y_true_list = np.asarray(y_true_list)

plt.plot(y_est_list[:, :, 0])
plt.plot(y_true_list[:, :, 0])
plt.show()
