import matplotlib.pyplot as plt
import numpy as np
import time


def slices_UV(U_star, V_star):
    U_slice_middle = np.zeros(V_star.shape)
    V_slice_middle = np.zeros(V_star.shape)
    U_slice_z5 = np.zeros(V_star.shape)
    V_slice_z5 = np.zeros(V_star.shape)
    pz = U_star.shape[2]
    for i in range(pz):
        U_slice_middle[:, :, i, :] = U_star[:, :, -1, :]
        V_slice_middle[:, :, i, :] = V_star[:, :, -1, :]
        U_slice_z5[:, :, i, :] = U_star[:, :, 9, :]
        V_slice_z5[:, :, i, :] = V_star[:, :, 9, :]
    return U_slice_middle, V_slice_middle, U_slice_z5, V_slice_z5


def trapeze(Z_star, Rho_star):
    z = np.linspace(Z_star.min(), Z_star.max(), Z_star.shape[2])
    py, px, pz = Rho_star.shape[0:3]
    z = z.squeeze()
    delta_zs = [z[i + 1] - z[i] for i in range(len(z) - 1)]
    Rho_star_ = np.zeros((py, px, Rho_star.shape[-1]))
    Dz = z.max() - z.min()
    for i in range(Rho_star.shape[-1]):
        Rho_star_[:, :, i] = (1 / Dz) * np.sum(
            [delta_zs[j] * (Rho_star[:, :, j + 1, i].reshape(py, px) + Rho_star[:, :, j + 1, i].reshape(py, px)) / 2 for
             j in range(len(z) - 1)], axis=0)
    Rho_mean = np.tile(Rho_star_[:, :, np.newaxis, :], (1, 1, pz, 1))
    return Rho_mean


def sample_data(path):
    dataset = np.load(path)

    X_star, Y_star, Z_star, T_star = [dataset[i,:] for i in range(4)]
    Rho_star, U_star, V_star = [dataset[i, :] for i in range(4,7)]


    Rho_mean = trapeze(Z_star, Rho_star)
    U_slice_middle, V_slice_middle, U_slice_z5, V_slice_z5 = slices_UV(U_star,V_star)





    ##preparing x_valid, y_valid
    x, y, z, t = [dataset[i,:].flatten().reshape(-1,1) for i in range(4)]
    rho, u, v, w, p = [dataset[i,:].flatten().reshape(-1,1) for i in range(4,9)]


    del(dataset)

    x_valid = np.concatenate([x, y, z, t],axis=1)
    y_valid = np.concatenate([rho, u, v, w , p],axis=1)

    ##x_train y_train
    x_train = np.copy(x_valid)
    rho_mean_train, u_mid_train, v_mid_train,\
        u_quart_train, v_quart_train = [ch.flatten().reshape(-1,1) for ch in
                                        [Rho_mean,
                                        U_slice_middle,
                                        V_slice_middle,
                                        U_slice_z5,
                                        V_slice_z5]
                                        ]


    y_train = np.concatenate([rho_mean_train, u_mid_train, v_mid_train, u_quart_train, v_quart_train, v_quart_train*0.0],axis=1)



    z_star = np.linspace(Z_star.min(),Z_star.max(),Z_star.shape[2]).reshape(-1,1)

    return (x_train,y_train),(x_valid,y_valid), z_star
