from absl import app
from absl import flags

import jax.numpy as jnp
import numpy as np

import jax
from jax import nn as jnn
from jax import random

from flax import nn
from flax import optim

import tensorflow_datasets as tfds


EPS = 1e-8
KEY = random.PRNGKey(0)

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The leanring rate for the Adam optimizer')
)

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training')
)

flags.DEFINE_integer(
    'num_epochs', default=10,
    help=('Number of training epochs')
)


class Encoder(nn.Module):
    def apply(self, x):
        x = nn.Dense(x, 400, name='enc_fc1')
        x = jnn.relu(x)
        mean_x = nn.Dense(x, 20, name='enc_fc21')
        logvar_x = nn.Dense(x, 20, name='enc_fc22')
        return mean_x, logvar_x


class Decoder(nn.Module):
    def apply(self, z):
        z = nn.Dense(z, 400, name='dec_fc1')
        z = jnn.relu(z)
        z = nn.Dense(z, 784, name='dec_fc2')
        z = jnn.sigmoid(z)
        return z


class VAE(nn.Module):
    def apply(self, x):
        mean, logvar = Encoder(x, name='encoder')
        z = reparameterize(mean, logvar)
        recon_x = Decoder(z, name='decoder')
        return recon_x, mean, logvar

    @nn.module_method
    def generate(self, z):
        params = self.get_param("decoder")
        return Decoder.call(params, z)


def reparameterize(mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = np.random.normal(size=logvar.shape)
    return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
    return - 0.5 * jnp.sum(1 + logvar - jnp.power(mean, 2) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy(probs, labels):
    return - jnp.sum(labels * jnp.log(probs + EPS) + (1 - labels) * jnp.log(1 - probs + EPS))


def compute_metrics(recon_x, x, mean, logvar):
    BCE = binary_cross_entropy(recon_x, x)
    KLD = kl_divergence(mean, logvar)
    return {'bce': jnp.mean(BCE), 'kld': jnp.mean(KLD), 'loss': jnp.mean(BCE + KLD)}


@jax.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        x = batch['image']
        recon_x, mean, logvar = model(x)

        BCE = binary_cross_entropy(recon_x, x)
        KLD = kl_divergence(mean, logvar)
        loss = jnp.mean(BCE + KLD)
        return loss, recon_x
    optimizer, _, _ = optimizer.optimize(loss_fn)
    return optimizer


@jax.jit
def eval(model, eval_ds, z):
    xs = eval_ds['image'] / 255.0
    xs = xs.reshape(-1, 784)
    recon_xs, mean, logvar = model(xs)

    comparison = jnp.concatenate([xs[:8].reshape(-1, 28, 28, 1),
                                  recon_xs[:8].reshape(-1, 28, 28, 1)])

    generate_xs = model.generate(z)
    generate_xs = generate_xs.reshape(-1, 28, 28, 1)

    return compute_metrics(recon_xs, xs, mean, logvar), comparison, generate_xs


def main(argv):
    train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
    train_ds = train_ds.cache().shuffle(1000).batch(FLAGS.batch_size)
    test_ds = tfds.as_numpy(tfds.load('mnist', split=tfds.Split.TEST, batch_size=-1))

    _, vae = VAE.create_by_shape(KEY, [((1, 784), jnp.float32)])

    optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(vae)

    for epoch in range(FLAGS.num_epochs):
        for batch in tfds.as_numpy(train_ds):
            batch['image'] = batch['image'].reshape(-1, 784) / 255.0
            optimizer = train_step(optimizer, batch)

        z = np.random.normal(size=(64, 20))
        metrics, comparison, sample = eval(optimizer.target, test_ds, z)

        print("eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}".format(
            epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
        ))


if __name__ == '__main__':
    app.run(main)