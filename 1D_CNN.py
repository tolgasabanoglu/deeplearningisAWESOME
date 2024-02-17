import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random

def regression_1d_enmap():
    
    def norm(a):
        a_norm = a.astype(np.float32)
        a_norm = a_norm / 10000
        return a_norm

    def get_model_1d(class_num):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(512, kernel_size=3, activation='relu', input_shape=(224, 1)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(class_num, activation='linear')  # Adjust activation based on the problem
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model
    
    lr = 1e-4
    batch_size = 64
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model_1d_enmap = get_model_1d(3)
    
    x_in = norm(np.load('/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Mixed_spectral/mixed_spectral_enmap.npy'))
    y_in = np.load('/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Mixed_spectral/label_enmap.npy')
    y_in = y_in.astype(np.float32)
    
    train_list = list(range(y_in.shape[0]))
    random.shuffle(train_list)
    
    iteration = int(y_in.shape[0] / batch_size)
    epochs = 3
    
    for e in range(epochs):
        loss_train = 0
        for it in tqdm(range(iteration)):
            train_batch = train_list[it * batch_size: it * batch_size + batch_size]
            x_batch = x_in[train_batch, ..., np.newaxis]  # Add a new axis for the channel
            y_batch = y_in[train_batch, ...]
            loss_train += model_1d_enmap.train_on_batch(x_batch, y_batch)
        
        loss_train = loss_train / iteration
        print('Enmap - MAE: ', loss_train)
        random.shuffle(train_list)
        
    tf.keras.models.save_model(model_1d_enmap, '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Saved_model/model_1D_enmap', save_format="tf")

regression_1d_enmap()
