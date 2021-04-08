import tensorflow as tf

def LSTMCell(hidden_dim, cell_type = 'LSTM'):
	cell = tf.keras.layers.LSTM(hidden_dim)



	return cell