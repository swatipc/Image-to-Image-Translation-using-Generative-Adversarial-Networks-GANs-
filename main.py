import os

from dataset import Dataset
from webapp.evaluate import Evaluator
from train import Train
from utils import Config

config = Config({
    'action': 'valid', # Change this to "train" to start training or "test" to generate one image and "valid" to run tests on all validation dataset
    'epochs': 100,
    'batchSize': 1,
    'trainDatasetFolder': 'datasets/maps/train/',
    'validDatasetFolder': 'datasets/maps/val/',
    'npzTrainDatasetPath': 'datasets/maps.npz',
    'npzValidDatasetPath': 'datasets/maps_valid.npz',
    'outputImageSize': (256, 512), # Training images are concatenated with validation images. So two squares images.
    'checkpointDir': 'checkpoints',
    'logsDir': 'logs/',

    'modelPath': 'checkpoints/models/model_032000.h5',
    'test_image': 'datasets/maps/train/'

})


def main():
    if config.action == "train":
        dataset = Dataset(config.trainDatasetFolder, config.npzTrainDatasetPath, config.outputImageSize)
        if os.path.exists(config.npzTrainDatasetPath):
            dataset.extract_train_target_images()
            print('Npz compressed dataset already saved at ', config.npzTrainDatasetPath)
        else:
            # Convert the raw image dataset into npz compressed format and extract
            dataset.process()
        # Plot a few sample images to verify data processing went well.
        dataset.plot_sample_input_target_images(5)

        trainer = Train(dataset, config.epochs, config.batchSize, config.checkpointDir, config.logsDir)
        trainer.start()
    elif config.action == "test":
        evaluator = Evaluator(config.modelPath)
        evaluator.predict(config.test_image)
    else:
        valid_dataset = Dataset(config.validDatasetFolder, config.npzValidDatasetPath, config.outputImageSize)
        if os.path.exists(config.npzValidDatasetPath):
            valid_dataset.extract_train_target_images()
            print('Npz compressed dataset already saved at ', config.npzTrainDatasetPath)
        else:
            # Convert the raw image dataset into npz compressed format and extract
            valid_dataset.process()
        evaluator = Evaluator(config.modelPath)
        evaluator.test_all_val_images(valid_dataset)


if __name__ == "__main__":
    main()
