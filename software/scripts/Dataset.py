import torch
from  torch.utils import data
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import random


class Dataset():

## -------------- Getters and Setters ------------
    def setBatchSize(self, size):
        self.batchSize = size

    def getBatchSize(self):
        return self.batchSize
    
    def getTrainsetArray(self):
        return self.trainsetArray

    def getTrainLabelArray(self):
        return self.trainLabelArray

    def getTestsetArray(self):
        return self.testsetArray
    
    def getTestLabelArray(self):
        return self.testLabelArray

    def getTrainsetSplit(self):
        return self.trainLoadedSplit
    
    def getValidatesetSplit(self):
        return self.validateLoadedSplit

## -------------- Other Functions ------------
    def downloadDataset(self):
         # Downloads the emnist dataset via pytorch
        trainDataset =  datasets.EMNIST(root='./emnist_data/',
                                        train = True, 
                                        split = 'byclass', 
                                         transform = transforms.Compose([
                                            lambda img: transforms.functional.rotate(img, -90), # rotate images 90 degress
                                            lambda img: transforms.functional.hflip(img),       # flip horizontally
                                            transforms.ToTensor()]), 
                                        download = True);

        testDataset = datasets.EMNIST(root='./emnist_data/',
                                        train = False, 
                                        split = 'byclass', 
                                        transform = transforms.Compose([
                                            lambda img: transforms.functional.rotate(img, -90), # rotate images 90 degress
                                            lambda img: transforms.functional.hflip(img),       # flip horizontally
                                            transforms.ToTensor()]));

        self.trainDataset = trainDataset;
        self.testDataset = testDataset;

    
    def loadSetsAndConvertToNumpy(self):
        # Loads the entire train and test set and converts to numpy array to be used to display on GUI
        # modified from https://stackoverflow.com/questions/54897646/pytorch-datasets-converting-entire-dataset-to-numpy
        print("Loading sets and converting to numpy arrays")
        # Loads trainset with a batch size of 10,000 and created a numpy array of images and labels
        trainAllLoader = DataLoader(self.trainDataset, batch_size=10000) # batch size is 10000, replace batch_size=100000 with
                                                                         # batch_size=len(self.trainDataset) if you want to 
                                                                         # you want to see the entire trainset (697,932 images)
                                                                         # (Note this will take an extremely long time to load 
                                                                         # when viewing train images)

        self.trainsetArray = next(iter(trainAllLoader))[0].numpy()
        self.trainLabelArray = next(iter(trainAllLoader))[1].numpy()
       
        # converting float values to rgb values of the image pixels values
        self.trainsetArray *= 255
        self.trainsetArray = self.trainsetArray.astype(np.uint8)
        print("Finished converting trainset to numpy array")

        # Loads testset with a batch size of 10,000 and created a numpy array of images and labels
        testAllLoader = DataLoader(self.testDataset, batch_size=10000) # batch size is 10000, replace batch_size=100000 with
                                                                       # batch_size=len(self.testDataset) if you want to 
                                                                       # see the entire testset (116,323 Images) 
                                                                       # (Note this will take an extremely long time to 
                                                                       # load when viewing test images)
        self.testsetArray = next(iter(testAllLoader))[0].numpy()
        self.testLabelArray = next(iter(testAllLoader))[1].numpy()

        # converting float values to rgb values of the image pixels values
        self.testsetArray *= 255
        self.testsetArray = self.testsetArray.astype(np.uint8)
        print("Finished converting testset to numpy array")
    
     
    def splitTrainAndValid(self,trainRatio):
     # Splits train dataset to train and validate sets depending on train/validate ratio
     # Paramaters: train ratio as a whole number ie: 75, 80 etc
     # Modified from: https://stackoverflow.com/questions/61623709/how-to-split-a-dataset-into-a-custom-training-set-and-a-custom-validation-set-wi

        # Get Train size from the train ratio
        trainSize = int(trainRatio/100 * len(self.trainDataset))

        indices = list(range(len(self.trainDataset)))
        random.seed(310)  # fix the seed so the shuffle will be the same everytime
        random.shuffle(indices)
        
        # Split the train dataset to train and validate
        trainDatasetSplit = torch.utils.data.Subset(self.trainDataset, indices[:trainSize])
        validateDatasetSplit = torch.utils.data.Subset(self.trainDataset, indices[trainSize:])

        # Load the split sets into a dataloader
        self.trainLoadedSplit = DataLoader(dataset = trainDatasetSplit,
                                            batch_size = self.batchSize,
                                            shuffle = True)
        self.validateLoadedSplit = DataLoader(dataset = validateDatasetSplit,
                                            batch_size = self.batchSize,
                                            shuffle = True)

