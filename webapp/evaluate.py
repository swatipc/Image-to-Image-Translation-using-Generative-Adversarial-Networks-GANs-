from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot
from numpy import vstack, expand_dims
from tensorflow.python.keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


# Normalize/Scale the pixel values to [-1,1] from [0, 255]
def scale_image(image_array):
    return (image_array - 127.5) / 127.5


def rescale(input_image_array):
    return (input_image_array + 1) / 2.0


class Evaluator:
    def __init__(self, model_path, is_cycle_gan = False):
        self.model_path = model_path
        if not is_cycle_gan:
            self.model = load_model(self.model_path)
        else:
            self.model = load_model(self.model_path, custom_objects={'InstanceNormalization': InstanceNormalization})

    def predict(self, input_img, image_size=(256, 256), plot=True):
        scaled_input_image = self.load_image(input_img, image_size)
        target_image = self.model.predict(scaled_input_image)
        if plot:
            self.plot_input_gen_images(scaled_input_image, target_image)
        else:
            return rescale(vstack(target_image))

    @staticmethod
    def load_image(filename, size=(256, 256)):
        image = load_img(filename, target_size=size)
        image_array = img_to_array(image)
        # Normalize/Scale the pixel values to [-1,1] from [0, 255]
        scaled_image = scale_image(image_array)
        return expand_dims(scaled_image, 0)

    # plot source, generated and target images
    @staticmethod
    def plot_input_gen_images(input_img, gen_img):
        images = vstack((input_img, gen_img))
        # scale back to [0,1] from [-1, 1]
        images = rescale(images)
        titles = ['Input', 'Generated']
        for i in range(len(images)):
            # define subplot
            pyplot.subplot(1, 2, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(images[i])
            # show title
            pyplot.title(titles[i])
        pyplot.show()

    # plot source, generated and target images
    @staticmethod
    def plot_input_gen_target_images(input_img, gen_img, target):
        images = vstack((input_img, gen_img, target))
        # scale back to [0,1] from [-1, 1]
        images = rescale(images)
        titles = ['Input', 'Generated', 'Target']
        for i in range(len(images)):
            # define subplot
            pyplot.subplot(1, 3, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(images[i])
            # show title
            pyplot.title(titles[i])
        pyplot.show()

    def test_all_val_images(self, valid_dataset):
        for i in range(len(valid_dataset.input_scaled)):
            input_image = expand_dims(valid_dataset.input_scaled[i], 0)
            target_image = expand_dims(valid_dataset.target_scaled[i], 0)
            generated_image = self.model.predict(input_image)
            self.plot_input_gen_target_images(input_image, generated_image, target_image)





