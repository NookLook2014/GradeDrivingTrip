from keras import backend as K
from keras import losses
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def deepAE(input_dim, num_hidden1, num_hidden2, encoding_dim, exist_weights_file=None):
    '''
    Build a 3-layer(sub) deep encoder-decoder network
    :param input_dim:
    :param num_hidden1:
    :param encoding_dim:
    :return:
    '''
    input_vec = Input(shape=(input_dim,))
    # add a Dense layer with a L1 activity regularizer
    encoded = Dense(num_hidden1, activation='relu',
                    kernel_initializer='normal',
                    activity_regularizer=regularizers.l1(10e-5))(input_vec)
    encoded = Dense(num_hidden2, activation='relu',
                    kernel_initializer='normal',
                    activity_regularizer=regularizers.l1(10e-5))(encoded)

    encoded = Dense(encoding_dim, activation='relu',
                    kernel_initializer='normal',
                    activity_regularizer=regularizers.l1(10e-5), name='encoder_output_layer')(encoded)

    decoded = Dense(num_hidden2, activation='relu',
                    kernel_initializer='normal')(encoded)
    decoded = Dense(num_hidden1, activation='relu',
                    kernel_initializer='normal',)(decoded)
    decoded = Dense(input_dim, activation='sigmoid',
                    kernel_initializer='normal')(decoded)

    autoencoder = Model(input_vec, decoded)

    if exist_weights_file:
        autoencoder.load_weights(exist_weights_file)

    encoder = Model(input_vec, encoded)
    # reconstruction_loss = mse(input_img, decoded)
    # reconstruction_loss *= input_dim
    # loss = K.mean(reconstruction_loss)
    # autoencoder.add_loss(loss)
    # optimizer = K.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-6)
    autoencoder.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=[root_mean_squared_error, 'mse', 'mae'])
    # print(encoder.summary())
    return autoencoder, encoder