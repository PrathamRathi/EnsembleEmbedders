import tensorflow as tf


class DenseVAE(tf.keras.Model):
    def __init__(self, instrument_units, pitch_units, song_length, learning_rate, latent_size=256, hidden_dim=256,epochs=1):
        super(DenseVAE, self).__init__()
        self.epochs = epochs
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.instrument_units = instrument_units
        self.pitch_units = pitch_units
        self.song_length = song_length

        input_shape = (self.instrument_units, self.pitch_units, self.song_length)
        flattened_dim = self.pitch_units * self.song_length * self.instrument_units
        if (instrument_units == 1):
            self.input_shape = (self.pitch_units, self.song_length)
            flattened_dim = self.pitch_units * self.song_length

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        ])
        self.mu_layer = tf.keras.layers.Dense(latent_size)
        self.logvar_layer = tf.keras.layers.Dense(latent_size)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(flattened_dim, activation='sigmoid'),
            tf.keras.layers.Reshape(input_shape)
        ])
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon. loss')
        self.kld_loss_tracker = tf.keras.metrics.Mean(name='kld loss')


    def get_latent_encoding(self, x):
        """
        Returns latent encoding of input
        Inputs:
        - x: a batch of input chroma
        Returns:
        - z: batch of latent encodings of input created by encoder and reparameterization
        """
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = self.reparametrize(mu, logvar)
        return z
    

    def call(self, x):
        """    
        Runs a forward pass of the entire vae
        Inputs:
        - x: Batch of input chroma of shape (N, 1, H, W)    
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z
    

    def predict(self, x):
        """
        Runs a forward pass on the data but only returns reconstructions
        Inputs:
        - x: Batch of input chroma of shape (N, 1, H, W)    
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        """
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat
    

    def reparametrize(self, mu, logvar):
        """
        Differentiably sample random Gaussian data with specified mean and variance using the
        reparameterization trick.

        Inputs:
        - mu: Tensor of shape (N, Z) giving means
        - logvar: Tensor of shape (N, Z) giving log-variances

        Returns: 
        - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
            mean mu[i, j] and log-variance logvar[i, j].
        """
        std_dev = tf.math.sqrt(tf.math.exp(logvar))
        z = mu + tf.random.normal(shape=tf.shape(std_dev)) * std_dev
        return z


    def bce_function(self,x_hat, x):
        """
        Computes the reconstruction loss of the VAE.
        
        Inputs:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - x: Input data for this timestep of shape (N, 1, H, W)
        
        Returns:
        - reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
        """
        bce_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.SUM,
        )
        reconstruction_loss = bce_fn(x, x_hat) * x.shape[
            -1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
        self.recon_loss_tracker.update_state(reconstruction_loss/x.shape[0])
        return reconstruction_loss


    def loss_function(self,x_hat, x, mu, logvar):
        """
        Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
        Returned loss is the average loss per sample in the current batch.

        Inputs:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - x: Input data for this timestep of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
        
        Returns:
        - loss: Tensor containing the scalar loss for the negative variational lowerbound
        """
        variance = tf.math.exp(logvar)
        kl_loss = -.5 * tf.math.reduce_sum((1 + logvar - tf.square(mu) - variance))
        self.kld_loss_tracker.update_state(kl_loss)
        loss = self.bce_function(x_hat, x) + kl_loss
        loss /= x.shape[0]
        return loss
    
    def train_step(self, data):
        x = data[0]
        with tf.GradientTape() as tape:
            x_hat, mu, logvar, _ = self.call(x)
            loss = self.loss_function(x_hat, x, mu, logvar)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {'loss':self.loss_tracker.result(),
                'recon. loss':self.recon_loss_tracker.result(),
                'kl loss':self.kld_loss_tracker.result()
                }