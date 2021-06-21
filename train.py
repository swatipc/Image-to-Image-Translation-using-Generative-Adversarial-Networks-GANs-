import os
from datetime import datetime

import tensorflow as tf

from matplotlib import pyplot
from tensorflow import zeros

from neural_network import Generator, Discriminator, GAN


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rescale(input_image_array):
    return (input_image_array + 1) / 2.0


class Train:

    def __init__(self, dataset, num_epochs, batch_size, checkpoint_dir, logs_dir):
        self.dataset = dataset
        self.train_dataset = dataset.input_scaled
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_batches = int(len(self.train_dataset) / batch_size)
        self.checkpoint_dir = checkpoint_dir
        # Tensorboard setup
        self.log_dir = logs_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.log_writer.set_as_default()

        # define input shape based on the loaded dataset
        self.image_shape = self.train_dataset.shape[1:]
        # Models
        self.generator_model = Generator(self.image_shape).model()
        self.discriminator_model = Discriminator(self.image_shape).model()

        # Build the GAN composite model using the above two models.
        self.gan_model = GAN(self.generator_model, self.discriminator_model, self.image_shape).model()

    def start(self):
        # Load Dataset
        print('Train Dataset Shape', self.train_dataset.shape)

        num_steps = self.num_batches * self.num_epochs
        patch_shape = self.discriminator_model.output_shape[1]
        for step in range(num_steps):
            # select a batch of real samples
            [X_real_sat_img, X_real_map_img], y_real = self.dataset.get_random_sample_input_batch(self.batch_size,
                                                                                                  patch_shape)
            # update discriminator for real samples
            d_loss1 = self.discriminator_model.train_on_batch([X_real_sat_img, X_real_map_img], y_real)

            # generate a batch of fake samples
            X_fake_map_img, y_fake = self.get_generated_fake_samples(X_real_sat_img, patch_shape)
            # update discriminator for generated samples
            d_loss2 = self.discriminator_model.train_on_batch([X_real_sat_img, X_fake_map_img], y_fake)
            # update the generator
            g_loss, _, _ = self.gan_model.train_on_batch(X_real_sat_img, [y_real, X_real_map_img])

            # Monitor Metrics
            self.monitor_metrics(step + 1, d_loss1, d_loss2, g_loss)

            # summarize model performance
            # We get nearly 50 checkpoints and we have choose the best one by looking the generated images manually.
            if (step + 1) % 2000 == 0:
                self.summarize_performance(step, 4)

    # generate a batch of images, returns images and targets
    def get_generated_fake_samples(self, samples, patch_shape):
        # generate fake instance
        X = self.generator_model.predict(samples)
        # create 'fake' class labels (0)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    # Save checkpoints and also save generated images vs real images for comparison
    def summarize_performance(self, step, n_samples):
        # select a sample of input images
        [X_real_sat_img, X_real_map_img], y_real = self.dataset.get_random_sample_input_batch(n_samples, 1)
        # generate a batch of fake samples
        X_fake_map_img, y_fake = self.get_generated_fake_samples(X_real_sat_img, 1)

        X_real_sat_img = rescale(X_real_sat_img)
        X_real_map_img = rescale(X_real_map_img)
        X_fake_map_img = rescale(X_fake_map_img)

        # plot actual satellite image
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_real_sat_img[i])

        # plot generated map image
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(X_fake_map_img[i])

        # plot actual target map image
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
            pyplot.axis('off')
            pyplot.imshow(X_real_map_img[i])

        # Create dirs and save models and figs
        create_dir(self.checkpoint_dir + "/images")
        create_dir(self.checkpoint_dir + "/models")

        image_path = self.checkpoint_dir + '/images/image_%06d.png' % (step + 1)
        pyplot.savefig(image_path)
        pyplot.close()
        print("Image saved at %s" % image_path)

        # save the generator model
        generator_model_path = self.checkpoint_dir + '/models/model_%06d.h5' % (step + 1)
        self.generator_model.save(generator_model_path)
        print("Model saved at %s" % generator_model_path)

    @staticmethod
    def monitor_metrics(step, d_loss1, d_loss2, g_loss):
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (step + 1, d_loss1, d_loss2, g_loss))
        tf.summary.scalar("d1_loss", d_loss1, step=step + 1)
        tf.summary.scalar("d2_loss", d_loss2, step=step + 1)
        tf.summary.scalar("g_loss", g_loss, step=step + 1)
