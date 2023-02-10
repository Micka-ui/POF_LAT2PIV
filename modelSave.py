import keras
import os
class modelSave(keras.callbacks.Callback):

    def __init__(self, frequence, path):
        super(modelSave, self).__init__()
        self.frequence = frequence
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequence == 0:
            self.model.save(os.path.join(self.path, 'model_%s' % (epoch)))