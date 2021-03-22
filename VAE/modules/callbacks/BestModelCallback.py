# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                        BestModelCallback
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 - S. Arias, E. Maldonado, JL. Parouty
# ------------------------------------------------------------------
# 2.0 version by JL Parouty, feb 2021

from tensorflow.keras.callbacks import Callback
import numpy as np
import os

                
class BestModelCallback(Callback):

    def __init__(self, filename= './run_dir/best-model.h5', verbose=0 ):
        self.filename = filename
        self.verbose  = verbose
        self.loss     = np.Inf
        os.makedirs( os.path.dirname(filename), mode=0o750, exist_ok=True)
                
    def on_train_begin(self, logs=None):
        self.loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if current < self.loss:
            self.loss = current
            self.model.save(self.filename)
            if self.verbose>0: print(f'Saved - loss={current:.6f}')
