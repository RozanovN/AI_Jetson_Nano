import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense, Masking
from keras.losses import sparse_categorical_crossentropy
from tqdm.keras import TqdmCallback

def custom_sparse_categorical_crossentropy(y_true, y_pred):
    mask_value = -1
    mask = tf.not_equal(y_true, mask_value)

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    loss = sparse_categorical_crossentropy(y_true_masked, y_pred_masked, from_logits=True)

    return loss
  

class GomokuPlayer:
  def __init__(self, board_size, num_epoch=10, batch_size=32):
    # CNN + RNN
    self.num_epoch = num_epoch
    self.batch_size = batch_size
    
    cnn = Sequential()
    cnn.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((3, 3), strides=(3, 3)))
    cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    
    rnn_input = Input(shape=(None, board_size, board_size, 1))
    masking = Masking(mask_value=0)(rnn_input)
    rnn_layer = TimeDistributed(cnn)(masking)
    rnn_layer = LSTM(128, return_sequences=True)(rnn_layer)
    rnn_output = Dense(board_size**2, activation='softmax')(rnn_layer)
    
    self.model = Model(inputs=rnn_input, outputs=rnn_output)
    
  def compile(self):
    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  
  def fit(self, dataset):
    self.model.fit(dataset, epochs=self.num_epoch, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    
  def evaluate(self, dataset):
    return self.model.evaluate(dataset)
    
  