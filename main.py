# This is a sample Python script.
import os

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PhysicsInformedNN import *
from load_data import sample_data
import time
from modelSave import *
import numpy as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
        path = 'data_3D.npy'
        (x_train,y_train), (x_valid,y_valid), z_star = sample_data(path)





        ##Flow Parameters
        try :
                physical_device = tf.config.list_physical_devices(device_type='GPU')
                tf.config.experimental.set_memory_growth(physical_device[0], True)

        except:
                print('No GPU')


        Re = 5500
        Sc = 1
        ReSc = Re * Sc

        path_log = 'models'
        dirname = time.strftime('Run_%m_%M_')
        name_dir = os.path.join(path_log, dirname)
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)
        pt_log = os.path.join(name_dir, 'loss.log')

        ###Callbacks for saving log
        csv_logger = tf.keras.callbacks.CSVLogger(pt_log, separator=',', append=True)

        ##Callback to save model each  frequence epoch
        frequence = 2
        call_model = modelSave(frequence, name_dir)

        ###PARAMETERS MODEL

        ##hidden_units
        num_layers = 8
        hidden_units = [200 for i in range(num_layers)]
        x_test = x_train[:3, :]

        ##num of epochs, and batch size
        EPOCHS = 300
        BATCH_SIZE = 4096

        ##LR_SCHEDULER
        lr_start = 5e-4
        lr_end = 1e-5
        print('Training starts at Learning_rate :%s and ends at Learning_rate : %s' % (lr_start, lr_end))
        Cst = (lr_end / lr_start) ** (1 / EPOCHS)
        funct = lambda epoch: lr_start * (Cst ** epoch)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(funct)

        ##All callbacks
        callbacks = [csv_logger, call_model, lr_schedule]
        ##Loss and optimizer
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        optimizer = tf.keras.optimizers.Adam()
        ##training metrics
        loss_rho = tf.keras.metrics.MeanAbsoluteError(name='loss_rho_mean')
        loss_u = tf.keras.metrics.MeanAbsoluteError(name='loss_u')
        loss_v = tf.keras.metrics.MeanAbsoluteError(name='loss_v')
        loss_w =  tf.keras.metrics.MeanAbsoluteError(name='loss_w')
        loss_e1 = tf.keras.metrics.MeanSquaredError(name='loss_e1')
        loss_e2 = tf.keras.metrics.MeanSquaredError(name='loss_e2')
        loss_e3 = tf.keras.metrics.MeanSquaredError(name='loss_e3')
        loss_e4 = tf.keras.metrics.MeanSquaredError(name='loss_e4')
        loss_e5 = tf.keras.metrics.MeanSquaredError(name='loss_e5')
        training_metrics = [loss_rho, loss_u, loss_v, loss_w, loss_e1, loss_e2, loss_e3, loss_e4, loss_e5]

        ##evaluation_metrics
        loss_rho_valid = tf.keras.metrics.MeanAbsoluteError(name='loss_rho_valid')
        loss_u_valid = tf.keras.metrics.MeanAbsoluteError(name='loss_u_valid')
        loss_v_valid = tf.keras.metrics.MeanAbsoluteError(name='loss_v_valid')
        loss_w_valid = tf.keras.metrics.MeanAbsoluteError(name='loss_w_valid')

        evaluation_metrics = [loss_rho_valid, loss_u_valid, loss_v_valid, loss_w_valid]
        model = PhysicsInformedNN(Re, Sc, hidden_units=hidden_units,
                                  x_test=x_test,
                                  z = z_star,
                                  loss_fn=loss_fn,
                                  data_loss_w=1,
                                  training_metrics=training_metrics,
                                  evaluation_metrics=evaluation_metrics)

        model.compile(loss=loss_fn, optimizer=optimizer)
        history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_valid, y_valid),
                  callbacks=callbacks)


