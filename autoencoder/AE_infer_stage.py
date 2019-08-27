from keras.layers import Input
from keras.models import Model
from keras.models import load_model

from autoencoder.AE_network import deepAE


def load_model_from_weights(data, file_path):
    input_dim = data.shape[1]
    num_hidden1 = 128
    encoding_dim = 10
    autoencoder,encoder = deepAE(input_dim, num_hidden1, encoding_dim, exist_weights_file=file_path)

    return autoencoder,encoder

def restore_encoder(file_path):
    encoder = load_model(file_path)
    return encoder

def load_model_from_file(data, file_path):
    input_dim = data.shape[1]
    num_hidden1 = 128
    encoding_dim = 10

    autoencoder = load_model(file_path)
    input_vec = Input(shape=(input_dim,))
    encoder = Model(input_vec, autoencoder.get_layer('encoder_output_layer'))

    return autoencoder, encoder


from tdrive_etl_utils import load_fuel_truck_dag_data
if __name__ == '__main__':
    freq_model_path = 'bak/encoder_ftruck_freq_dag.model'
    model_path = 'bak/encoder_ftruck_time_dag.model' # [0.01334081595472155, 0.0004610253847615202, 0.002220902523304476]
    data = load_fuel_truck_dag_data()#.iloc[:,:9*9*1+1]
    vid = 54
    print(data.shape)
    data = data.tail(20)
    freq_encoder = restore_encoder(freq_model_path)
    res = freq_encoder.predict(data.iloc[:,1:])
    print(res)
    time_encoder = restore_encoder(model_path)
    # for _, row in data.iterrows():
    #     print(row.shape)
    #     print(row[1:].shape)
    #     # print(row[1:-1])
    #     reduced_vector = encoder.predict(row[1:])
    #     print(reduced_vector)


