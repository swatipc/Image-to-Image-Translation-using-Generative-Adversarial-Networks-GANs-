from os import listdir

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from numpy import savez_compressed, load, ones
from numpy.random import randint


# Normalize/Scale the pixel values to [-1,1] from [0, 255]
def scale_image(image_array):
    return (image_array - 127.5) / 127.5


class Dataset:
    def __init__(self, input_folder, npz_path, output_image_size):
        self.input_folder = input_folder
        self.npz_path = npz_path
        self.output_image_size = output_image_size
        self.input = None
        self.target = None
        self.input_scaled = None
        self.target_scaled = None

    def convert_raw_to_array(self):
        input_images = []
        target_images = []

        for filename in listdir(self.input_folder):
            # Change the size of the image to the output size
            pil_image = load_img(self.input_folder + filename, target_size=self.output_image_size)
            image_array = img_to_array(pil_image)
            # As the source image and target image are concatenated side by side in the input image,
            # we need to divide it into input and target images by splitting in the middle.
            # Input Satellite images
            input_images.append(image_array[:, :256])
            # Target Map images
            target_images.append(image_array[:, 256:])

        return [input_images, target_images]

    # Method to read dataset in raw jpg format and convert it to compressed npz format.
    def convert_to_npz(self):
        [input_images_array, target_images_array] = self.convert_raw_to_array()
        savez_compressed(self.npz_path, input_images_array, target_images_array)
        print('Compressed dataset saved at ', self.npz_path)

    # Method to read npz dataset, extract and normalize training images
    def extract_train_target_images(self):
        data = load(self.npz_path)
        # Extract train and val images datasets
        self.input, self.target = data['arr_0'], data['arr_1']
        # Normalize the pixel values to [-1,1] from [0, 255]
        self.input_scaled = scale_image(self.input)
        self.target_scaled = scale_image(self.target)

    def process(self):
        # Convert raw images to compressed numpy (npz) format
        self.convert_to_npz()
        # Extract and normalize images from npz dataset
        self.extract_train_target_images()

    def plot_sample_input_target_images(self, num_samples):
        for i in range(num_samples):
            pyplot.subplot(2, num_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(self.input[i].astype('uint8'))
        # plot target image
        for i in range(num_samples):
            pyplot.subplot(2, num_samples, 1 + num_samples + i)
            pyplot.axis('off')
            pyplot.imshow(self.target[i].astype('uint8'))
        pyplot.show()

    # Returns a batch of random samples
    def get_random_sample_input_batch(self, num_samples, patch_shape):
        # choose random instances
        index = randint(0, self.input_scaled.shape[0], num_samples)
        input_sat, target_map = self.input_scaled[index], self.target_scaled[index]
        # Fill real y with all ones
        y = ones((num_samples, patch_shape, patch_shape, 1))
        return [input_sat, target_map], y


