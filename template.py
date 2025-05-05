import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd

# 1. Data Loading & Preprocessing (Illustrative)
def load_and_preprocess_data(filepath, sequence_length):
    """Loads CSV, preprocesses, and creates sequences."""
    # Load your data from CSV or another source
    df = pd.read_csv(filepath)  # Replace with your data loading
    # Ensure 'Time' is datetime and set as index
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')
    # Select Features (example)
    features = ['Last', 'Vol']  # Add more features
    df = df[features].copy()
    # Data Scaling (Normalization)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    # Create Sequences (sliding window)
    X = []
    for i in range(len(df) - sequence_length):
        X.append(df[i:(i+sequence_length)].values)
    X = np.array(X)
    return X, scaler # Return the scaler for later use

# 2. LSTM-VAE Model Definition
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_lstm_vae(sequence_length, num_features, latent_dim):
    """Defines the LSTM-VAE model."""
    # --- Encoder ---
    encoder_inputs = tf.keras.Input(shape=(sequence_length, num_features))
    lstm_enc = layers.LSTM(64, return_sequences=True, activation='relu')(encoder_inputs) # You may need to adapt number of neurons
    lstm_enc = layers.LSTM(32, return_sequences=False, activation='relu')(lstm_enc)
    z_mean = layers.Dense(latent_dim, name="z_mean")(lstm_enc)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(lstm_enc)
    z = Sampling()([z_mean, z_log_var]) #Latent space

    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # --- Decoder ---
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    #Expand to match sequence_length
    repeat_vector = layers.RepeatVector(sequence_length)(latent_inputs)
    lstm_dec = layers.LSTM(32, return_sequences=True, activation='relu')(repeat_vector)
    lstm_dec = layers.LSTM(64, return_sequences=True, activation='relu')(lstm_dec) #Adapt the layer architecture
    decoder_outputs = layers.TimeDistributed(layers.Dense(num_features))(lstm_dec) #Reconstructing number of features

    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

    #Full VAE
    z_mean, z_log_var, z = encoder(encoder_inputs)
    decoder_outputs = decoder(z)
    vae = models.Model(encoder_inputs, decoder_outputs, name='vae')

    # VAE Loss - Reconstruction Loss + KL Divergence
    reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - decoder_outputs))
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(kl_loss)

    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    return vae, encoder, decoder

# 3. Training
def train_vae(vae, X_train, epochs, batch_size):
    """Trains the LSTM-VAE."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Adapt learning rate as needed
    vae.compile(optimizer=optimizer)
    vae.fit(X_train, epochs=epochs, batch_size=batch_size)

# 4. Anomaly Scoring
def calculate_anomaly_score(vae, encoder, x, scaler):
    """Calculates anomaly score for a single sequence."""
    x_scaled = scaler.transform(x)  # Scaling the data
    x_input = np.array([x_scaled]) # Reshape for single input

    z_mean, z_log_var, z = encoder.predict(x_input)
    reconstructed_x = vae.predict(x_input)
    reconstruction_loss = np.mean(np.square(x_input - reconstructed_x))

    kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))

    anomaly_score = reconstruction_loss + kl_loss
    return anomaly_score

# 5. Real-Time Anomaly Detection (Conceptual)
def real_time_anomaly_detection(data_stream, vae, encoder, scaler, sequence_length, anomaly_threshold):
    """Processes incoming data and flags anomalies."""
    # data_stream is assumed to be a continuous feed of new stock data
    sequence_buffer = []  # To hold the last 'sequence_length' data points
    for data_point in data_stream: # data_point is a single record of new stock data
        sequence_buffer.append(data_point)
        if len(sequence_buffer) >= sequence_length:
            # Transform sequence_buffer into the format that your model expect
            x = np.array(sequence_buffer[-sequence_length:]) # Take the last sequence
            anomaly_score = calculate_anomaly_score(vae, encoder, x, scaler) # Calculate the anomaly score
            if anomaly_score > anomaly_threshold:
                print("Anomaly Detected! Score:", anomaly_score)
                # Add your alerting/logging mechanism here

# ----- Main Execution -----
if __name__ == "__main__":
    # 1. Define Parameters
    sequence_length = 60  # Window of 60 minutes for LSTM
    num_features = 2    # 'Last' price and 'Vol'  - Adapt this to your number of features
    latent_dim = 16     # Latent space dimension
    epochs = 10
    batch_size = 32
    anomaly_threshold = 0.5  # Adjust this based on your data

    # 2. Load and Preprocess Data
    X_train, scaler = load_and_preprocess_data("your_training_data.csv", sequence_length)

    # 3. Build the LSTM-VAE Model
    vae, encoder, decoder = build_lstm_vae(sequence_length, num_features, latent_dim)

    # 4. Train the Model
    train_vae(vae, X_train, epochs, batch_size)

    # 5. Real-Time Anomaly Detection Simulation
    # (Replace this with your real-time data feed)
    # Dummy data stream - replace with a connector to your actual data source
    class DummyDataStream:
        def __init__(self, file_path, sequence_length):
            self.df = pd.read_csv(file_path) # Again, adapt this to your needs
            self.sequence_length = sequence_length
            self.index = sequence_length # start after the length of sequence

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.df):
                 raise StopIteration
            row = self.df.iloc[self.index].values #get each row of data
            self.index += 1
            return row

    dummy_data_stream = DummyDataStream("your_realtime_data.csv", sequence_length)
    real_time_anomaly_detection(dummy_data_stream, vae, encoder, scaler, sequence_length, anomaly_threshold)