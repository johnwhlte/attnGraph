import numpy as np
import torch
import math
import matplotlib.pyplot as plt

def function1x(x, y, t, alpha):
    return -np.exp(-alpha * t)*(np.exp(x)*np.cos(y))
def function1y(x, y, t, alpha):
    return np.exp(-alpha * t)*(np.exp(x)*np.sin(y))
def function2x(x, y, t, alpha):#w(z) = sin(z) dwdz=cos(z)
    return -(alpha * t)*(np.cos(alpha*t*x)*np.cosh(alpha*t*y))
def function2y(x, y, t, alpha):
    return (alpha * t)*(-np.sin(alpha*t*x)*np.sinh(alpha*t*y))
def function3x(x, y, t, alpha):#w(z) = cos(z) dwdz = - sin(z)
    return (alpha * t)*(np.sin(alpha * t*x)*np.cosh(alpha * t*y))
def function3y(x, y, t, alpha):
    return (-alpha * t)*(-np.sinh(alpha * t*y)*np.cos(alpha * t*x))
def function4x(x, y, t, alpha):#w(z) = e^z dwdz = e^z
    return -alpha*t*(np.exp(alpha*x*t)*np.cos(y*t))
def function4y(x, y, t, alpha):
    return alpha * t*(np.exp(alpha*x*t)*np.sin(y*t))

def define_soln(vel_funcs, time_vector, time_step, domain_x, domain_y, split_x, split_y, alpha):

    time_range = np.linspace(time_vector[0], time_vector[1], math.floor(time_vector[1] / time_step))
    x_range = np.linspace(domain_x[0], domain_x[1], math.floor(domain_x[1] / split_x))
    y_range = np.linspace(domain_y[0], domain_y[1], math.floor(domain_y[1] / split_y))
    y, x = np.meshgrid(x_range, y_range)
    list_of_solns = []

    for t in time_range:
        print(t)
        #solution_tensor = torch.ones((5,))
        soln_tensor_vx = vel_funcs[0](x,y,t,alpha)
        soln_tensor_vy = vel_funcs[1](x,y,t,alpha)
        # for x in x_range:
        #     for y in y_range:
        #         soln = torch.tensor([x, y, t, vel_funcs[0](x, y, t, alpha), vel_funcs[1](x, y, t, alpha)])
        #         solution_tensor = torch.cat((solution_tensor, soln), dim=-1)

        # soln = solution_tensor[5:]
        alpha_array = np.full_like(x.flatten(), alpha)
        list_of_solns.append(np.asarray([x.flatten(), y.flatten(), alpha_array, soln_tensor_vx.flatten(), soln_tensor_vy.flatten()]).T)



    return list_of_solns


if __name__ == "__main__":

    function_list = [[function4x,function4y],[function2x,function2y],[function3x, function3y]]
    func_names = ["func4","func2","func3"]
    alpha_list = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09]
    times = []
    vals = []
    for i, funcs in enumerate(function_list):

        for alpha in alpha_list:

            soln_list = define_soln(funcs, time_vector=(0,10), time_step=0.01, domain_x=(0,1), domain_y=(0,1), split_x=0.05, split_y=0.05, alpha=alpha)
            for j, snapshot in enumerate(soln_list):

                print(snapshot.shape)

                torch.save(torch.tensor(snapshot,dtype=torch.float32),f'/home/sysiphus/bigData/snapshots/{func_names[i]}_{j}_{alpha}_snap.pt')
            # times.append(j)
            # vals.append(snapshot[20][3])

                if alpha == 0.17 and j % 10 == 0:
                    x = np.reshape(snapshot.T[0].T, (20,20))
                    y = np.reshape(snapshot.T[1].T, (20,20))
                    vy = np.reshape(snapshot.T[4].T, (20,20))
                    vx = np.reshape(snapshot.T[3].T, (20,20))

                    fig, ax = plt.subplots()
                    c = ax.pcolormesh(x,y,vx, cmap='RdBu')#, vmin=-z.max(), vmax=z.max())
                    ax.set_title('pcolormesh')
                    ax.axis([x.min(), x.max(), y.min(), y.max()])
                    fig.colorbar(c, ax=ax)
                    plt.savefig(f'../vis/{func_names[i]}check{alpha}_{j}_.png')


#       fig = plt.figure()
#        plt.scatter(times, vals)\
        # x = snapshot[0:20]
        # y = snapshot[20:40]
        # z = snapshot[40:60]
        # z2 = snapshot[60:]

        # fig, ax = plt.subplots()
        # c = ax.pcolormesh(x,y,z2, cmap='RdBu')#, vmin=-z.max(), vmax=z.max())
        # ax.set_title('pcolormesh')
        # ax.axis([x.min(), x.max(), y.min(), y.max()])
        # fig.colorbar(c, ax=ax)
        # plt.savefig(f'check2{alpha}_{i}_.png')
        # fig, ax = plt.subplots()
        # c = ax.pcolormesh(x,y,np.sqrt(z**2 + z2**2), cmap='RdBu')#, vmin=-z.max(), vmax=z.max())
        # ax.set_title('pcolormesh')
        # ax.axis([x.min(), x.max(), y.min(), y.max()])
        # fig.colorbar(c, ax=ax)
        # plt.savefig(f'check3{alpha}_{i}_.png')
