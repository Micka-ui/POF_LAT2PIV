import tensorflow as tf
import keras


class PhysicsInformedNN(keras.Model):
    def __init__(self, Re, Sc, hidden_units, x_test, z, loss_fn, data_loss_w=100, training_metrics=[],
                 evaluation_metrics=[]):
        super(PhysicsInformedNN, self).__init__()
        ##Reynold number and Schmidt number
        self.data_loss_w = data_loss_w
        self.loss_fn = loss_fn
        self.Re = Re
        self.Sc = Sc
        self.z = z
        self.ReSc = self.Re * self.Sc
        self.training_metrics = training_metrics
        self.evaluation_metrics = evaluation_metrics
        self.list_layers = [tf.keras.layers.Dense(hidden_units[0], activation='swish', input_shape=[x_test.shape[1]])] + \
                           [tf.keras.layers.Dense(hidden_unit,activation='swish') for hidden_unit in hidden_units] + \
                           [tf.keras.layers.Dense(5)]
        self.model_eqns = tf.keras.Sequential(self.list_layers)
        test = self.model_eqns(x_test)

    def __call__(self, inputs,training=True):
        pred = self.model_eqns(inputs, training=training)
        rho, u, v, w ,p = [tf.reshape(pred[:, i], (-1, 1)) for i in range(5)]
        return rho, u, v, w, p

    def train_step(self, data):
        x_train, y_train = data

        # unpack x_train in x,y,t
        x, y, z, t = [tf.reshape(x_train[:, i], (-1, 1)) for i in range(4)]
        z_mid = self.z[-1]*tf.ones(tf.shape(x))
        z_quart = self.z[9]*tf.ones(tf.shape(x))

        # data obs for rho and data obs for the equations
        rho_mean_train, u_mid_train,\
            v_mid_train, u_quart_train,\
            v_quart_train, y_eqns = [tf.reshape(y_train[:, i], (-1, 1)) for i in range(6)]
        w_mid_train = y_eqns

        with tf.GradientTape() as tape:
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(y)
                t2.watch(z)
                t2.watch(t)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(x)
                    t1.watch(y)
                    t1.watch(z)
                    t1.watch(t)
                    X = tf.stack([x[:, 0], y[:, 0], z[:,0], t[:, 0]], axis=1)
                    X_mid = tf.stack([x[:, 0], y[:, 0], z_mid[:, 0], t[:, 0]], axis=1)
                    X_quart = tf.stack([x[:, 0], y[:, 0], z_quart[:, 0], t[:, 0]], axis=1)
                    rho, u, v, w, p = self(X)
                    _, u_mid, v_mid, w_mid, _ = self(X_mid)
                    _, u_quart, v_quart, _, _ = self(X_quart)
                    rho_mean = self.trapeze(X)

                ##rho 1st derivatives
                rho_x, rho_y, rho_z, rho_t = [t1.gradient(rho, var) for var in [x, y, z, t]]
                ##u 1st derivatives
                u_x, u_y, u_z, u_t = [t1.gradient(u, var) for var in [x, y, z, t]]
                ##v 1st derivatives
                v_x, v_y, v_z, v_t = [t1.gradient(v, var) for var in [x, y, z, t]]
                ##w 1st derivatives
                w_x, w_y, w_z, w_t = [t1.gradient(w, var) for var in [x, y, z, t]]
                ##p 1st derivatives
                p_x, p_y, p_z = [t1.gradient(p, var) for var in [x, y, z]]
            ##second derivatoves
            rho_xx, rho_yy, rho_zz,\
            u_xx, u_yy, u_zz,\
            v_xx, v_yy, v_zz,\
            w_xx, w_yy, w_zz = [t2.gradient(*ij) for ij in
                                                      zip([rho_x, rho_y, rho_z,
                                                           u_x, u_y, u_z,
                                                           v_x, v_y, v_z,
                                                           w_x, w_y, w_z], [x, y, z] * 4)]

            e1 = (u_t + u * u_x + v * u_y + w * u_z) + p_x - (1 / self.Re) * (u_xx + u_yy + u_zz)
            e2 = (v_t + u * v_x + v * v_y + w * v_z) + p_y - (1 / self.Re) * (v_xx + v_yy + v_zz) + rho
            e3 = (w_t + u * w_x + v * w_y + w * w_z) + p_z - (1 / self.Re) * (w_xx + w_yy + w_zz)
            e4 = u_x + v_y + w_z
            e5 = (rho_t + u * rho_x + v * rho_y + w * rho_z) - (1 / self.ReSc) * (rho_xx + rho_yy + rho_zz)

            square_eqns = [tf.square(e_i) for e_i in [e1, e2, e3, e4, e5]]

            square_rho = tf.square(rho_mean_train - rho_mean)
            square_u = tf.square(u_mid_train - u_mid) + tf.square(u_quart_train - u_quart)
            square_v = tf.square(v_mid_train - v_mid) + tf.square(v_quart_train - v_quart)
            square_w = tf.square(w_mid_train - w_mid)


            for e_i, metric in zip([square_rho, square_u, square_v, square_w, e1, e2, e3, e4, e5], self.training_metrics):
                metric.update_state(e_i, y_eqns)

            loss_eqns = self.loss_fn(tf.reduce_sum(square_eqns), y_eqns)
            loss_obs = self.loss_fn(tf.reduce_sum([square_rho,square_u, square_v, square_w]), y_eqns)
            loss = self.data_loss_w * loss_obs + loss_eqns

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        metrics = self.training_metrics
        return {m.name: m.result() for m in metrics}

    def trapeze(self, inputs):
        x = tf.reshape(inputs[:, 0], (-1, 1))
        y = tf.reshape(inputs[:, 1], (-1, 1))
        t = tf.reshape(inputs[:, 3], (-1, 1))
        z_tem = tf.ones(tf.shape(x))

        list_rho = []
        z = self.z
        delta_zs = [z[i + 1] - z[i] for i in range(len(z) - 1)]
        for ind, zi in enumerate(z):
            X = tf.stack([x[:, 0], y[:, 0], (z_tem * zi)[:, 0], t[:, 0]], axis=1)
            Y = self.model_eqns(X, training=True)
            list_rho.append(tf.reshape(Y[:, 0], (-1, 1)))

        mean_rho = tf.reduce_sum([(delta_zs[i] / 2) * (list_rho[i + 1] + list_rho[i]) for i in range(len(z) - 1)],
                                 axis=0)
        Dz = z.max() - z.min()
        mean_rho = (1/Dz) * tf.reshape(mean_rho, (-1, 1))
        return mean_rho

    def test_step(self, data):
        x, y = data

        rho_valid, u_valid, v_valid, w_valid = [tf.reshape(y[:, i], (-1, 1)) for i in range(4)]

        # Compute predictions
        rho_pred, u_pred, v_pred, w_pred , _ = self(x, training=False)

        for metric, valid, pred in zip(self.evaluation_metrics, [rho_valid, u_valid, v_valid, w_valid],
                                       [rho_pred, u_pred, v_pred, w_pred]):
            metric.update_state(valid, pred)

        # Updates the metrics tracking the loss

        metrics = self.evaluation_metrics

        return {m.name: m.result() for m in metrics}