import tensorflow
from keras import backend as K

# Query ciphertext the main server saved (the client's encrypted feature vector).
SHARE_PATH = '/workspace/shared_data/ciphertexts/target_enc'
# Where this cluster writes its encrypted partial result (main reads it back).
CLUSTERING_1 = '/workspace/shared_data/results/clustering_1'

# Registered/enrolled ciphertext for this cluster, produced by the client.
SAVED_CTXT1 = '/workspace/shared_data/ciphertexts/ctxt1'

PK_PATH = '/workspace/shared_data/keys/public.key'
GK_PATH = '/workspace/shared_data/keys/galois.key'
RL_PATH = '/workspace/shared_data/keys/relin.key'
FEATURE_MODEL_PATH = '/workspace/shared_data/models/feature_model'
MODEL_PATH = '/workspace/shared_data/models/model'

CLUSTER_NUM = 1 

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

swish = tensorflow.keras.activations.swish
