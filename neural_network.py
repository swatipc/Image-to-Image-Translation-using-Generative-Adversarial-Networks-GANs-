from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Concatenate, LeakyReLU, BatchNormalization, Activation, \
    Conv2DTranspose, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.ops.init_ops import RandomNormal


class Discriminator:
    def __init__(self, image_shape=(256, 256, 3)):
        self.image_shape = image_shape
        self.init = RandomNormal(stddev=0.02)
        self.input_image = Input(image_shape)
        self.target_image = Input(image_shape)
        self.merged_image = Concatenate()([self.input_image, self.target_image])

        # Layers
        self.layer_1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)
        self.layer_2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)
        self.layer_3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)
        self.layer_4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)
        self.layer_5 = Conv2D(512, (4, 4), padding='same', kernel_initializer=self.init)
        self.layer_6 = Conv2D(1, (4, 4), padding='same', kernel_initializer=self.init)

    def model(self):
        out = LeakyReLU(0.2)(self.layer_1(self.merged_image))
        out = LeakyReLU(0.2)(BatchNormalization()(self.layer_2(out)))
        out = LeakyReLU(0.2)(BatchNormalization()(self.layer_3(out)))
        out = LeakyReLU(0.2)(BatchNormalization()(self.layer_4(out)))
        out = LeakyReLU(0.2)(BatchNormalization()(self.layer_5(out)))

        patch_out = Activation('sigmoid')(self.layer_6(out))
        model = Model([self.input_image, self.target_image], patch_out)

        # Using Adam optimizer
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])
        print("\n************************************* Discriminator Model ****************************************")
        print(model.summary())
        plot_model(model, "modelplots/pix2pix/discriminator_model.png", show_shapes=True, show_layer_names=True)
        return model


class Generator:
    def __init__(self, image_shape=(256, 256, 3)):
        self.image_shape = image_shape
        self.init = RandomNormal(stddev=0.02)
        self.input_image = Input(image_shape)

    def encoder(self, input_layer, num_filters, batch_norm=True):
        # Down sampling the layer
        out = Conv2D(num_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(input_layer)
        # Add batch normalization if exists
        if batch_norm:
            out = BatchNormalization()(out, training=True)
        # leaky relu activation
        return LeakyReLU(alpha=0.2)(out)

    def decoder(self, input_layer, skip_layer, num_filters, dropout=True):
        # Up sampling the layer
        out = Conv2DTranspose(num_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(input_layer)
        out = BatchNormalization()(out, training=True)
        # Dropout is exists
        if dropout:
            out = Dropout(0.5)(out, training=True)
        # Merge skip layers
        out = Concatenate()([out, skip_layer])
        # leaky relu activation
        return Activation('relu')(out)

    def model(self):
        down1 = self.encoder(self.input_image, 64, batch_norm=False)
        down2 = self.encoder(down1, 128)
        down3 = self.encoder(down2, 256)
        down4 = self.encoder(down3, 512)
        down5 = self.encoder(down4, 512)
        down6 = self.encoder(down5, 512)
        down7 = self.encoder(down6, 512)
        # Not adding batch normalization and Relu to the bottle neck layer
        bottleneck = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(down7)
        bottleneck = Activation('relu')(bottleneck)
        # decoder model
        up1 = self.decoder(bottleneck, down7, 512)
        up2 = self.decoder(up1, down6, 512)
        up3 = self.decoder(up2, down5, 512)
        up4 = self.decoder(up3, down4, 512, dropout=False)
        up5 = self.decoder(up4, down3, 256, dropout=False)
        up6 = self.decoder(up5, down2, 128, dropout=False)
        up7 = self.decoder(up6, down1, 64, dropout=False)
        # output
        out = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.init)(up7)
        out_image = Activation('tanh')(out)
        # define model
        model = Model(self.input_image, out_image)
        print("\n**************************************** Generator Model *****************************************")
        print(model.summary())
        plot_model(model, "modelplots/pix2pix/generator_model.png", show_shapes=True, show_layer_names=True)
        return model


class GAN:
    def __init__(self, generator, discriminator, image_shape=(256, 256, 3)):
        self.generator = generator
        self.discriminator = discriminator
        self.input_image = Input(image_shape)

    def model(self):
        # Don't train the discriminator model weights
        self.discriminator.trainable = False

        # Send the image to the generator model
        generator_output = self.generator(self.input_image)

        # Send the actual input and generator output to discriminator model
        discriminator_out = self.discriminator([self.input_image, generator_output])

        #  Final Model
        model = Model(self.input_image, [discriminator_out, generator_output])
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1, 100])
        print("\n******************************************* GAN Model ********************************************")
        print(model.summary())
        plot_model(model, "modelplots/pix2pix/gan.png", show_shapes=True, show_layer_names=True)
        return model
