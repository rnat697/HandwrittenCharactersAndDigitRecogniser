
import torch
from torch import nn, optim, cuda
import numpy as np
import torch.nn.functional as F

# Class handles the training, testing and predictions of DNN models
# training, testing and save trained model functions modifted from: 
# https://github.com/Ti-Oluwanimi/Neural-Network-Classification-Algorithms/blob/main/AlexNet.ipynb
class TrainOrTestModel():
    def __init__(self,epoch):
        super().__init__()
        # Pass in epoch number
        self.epochNum = epoch
        self.trainProgress = 0
        self.validateProgress = 0
        self.averageLoss = 0
        self.trainAccuracy = 0
        self.currentEpoch = 0
        self.drawingAccuracy = 0

    # ----- Getters ------
    def getTrainingProgress(self):
        return self.trainProgress
    
    def getValidateProgress(self):
        return self.validateProgress
    def getAverageLoss(self):
        return self.averageLoss
    def getTrainAccuracy(self):
        return self.trainAccuracy
    def getMaxEpoch(self):
        return self.epochNum
    def getCurrentEpoch(self):
        return self.currentEpoch
    def getDrawingAccuracy(self):
        return self.drawingAccuracy
    def getSelectedImagesAccuracy(self):
        return self.selectImagesAccuracy
    def getPredictedClasses(self):
        return self.predictedClassesArray

# ------ Training, Testing and Predicting functions ------
    def training(self,model,savedModelName, trainDL, validDL):
        # This trains a given model with images from the train set and validates it with the validate set
        # Paramaters: model class (ie: LeNet5() or SpinalNet()), a trained model name, traindataload and validatedataload
        # ----- prep for training ----
        device = torch.device('cuda' if cuda.is_available() else 'cpu') #training with either cpu or cuda
        print("device is {}".format(device))
        modelTotrain = model
        modelToTrain = modelTotrain.to(device=device) #to send the model for training on either cuda or cpu
        modelToTrain.eval()
        
        ## Loss and optimizer
        learning_rate = 1e-4
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(modelToTrain.parameters(), lr= learning_rate)

        # ---- Training ----
        for epoch in range(self.epochNum): 
            loss_ep = 0
            self.currentEpoch = epoch
            for batch_idx, (data, targets) in enumerate(trainDL):
                data = data.to(device=device)
                targets = targets.to(device=device)
                optimizer.zero_grad()
                scores = modelToTrain(data)
                loss = criterion(scores,targets)
                loss.backward()
                optimizer.step()
                loss_ep += loss.item()
            
                # Calculate training progress
                currentNumImages = batch_idx*len(data)
                totalNumImages =len(trainDL.dataset)
                batchprogress = (currentNumImages/totalNumImages)*100 # amount of images in batch / total images with epoch
                self.trainProgress = round(batchprogress)

            # Calculate total loss
            self.averageLoss += loss_ep/len(trainDL)
            print(f"Loss in epoch {epoch} :::: {loss_ep/len(trainDL)}")
        
        # calculate average loss
        self.averageLoss = round(self.averageLoss / self.epochNum,2)
        print("Average Loss: {}".format(self.averageLoss))

        # ---- Validate ----
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data,targets) in enumerate(validDL):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = modelToTrain(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

                self.validateProgress = round(((batch_idx+1)/len(validDL))*100)
            
            # Calculate accuracy in XX.XX% format
            self.trainAccuracy = round(float(num_correct) / float(num_samples) * 100, 2)
            print(f"Got {num_correct} / {num_samples} with accuracy {self.trainAccuracy}")

            # Save model to a .pt file
            fileName = "../software/"+ savedModelName + ".pt"
            torch.save(modelToTrain.state_dict(), fileName) 
            print("Model saved!")
            
    
    
    def testing(self,model,imagesArray, labelsArray, filepath):
        # The model predicts the class (ie: what digit or letter it is) from an array of images from the test set
        # Paramaters: model class (ie: LeNet5() or SpinalNet()), the image numpy array, the labels numpy array and 
        # file path of saved model
        with torch.no_grad():
            trainedModel = model
            trainedModel.load_state_dict(torch.load(filepath)) #loads the trained model
            trainedModel.eval()
            self.predictedClassesArray= []
            charArray = ["0","1","2","3","4","5","6","7","8","9",
                        "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                        "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
            num_correct = 0
            num_samples = 0
            i = 0
            for image in imagesArray:
                # Convert image to tensor and change the pixel values from 0-255 to floats
                image = image.astype(np.float32)
                image = torch.from_numpy(image)# Convert to tensor 
                image = torch.unsqueeze(image, dim = 0)/255. # Add another dimension and map the gray scale to (between 0 and 1) 
                
                prediction = trainedModel(image)

                # Predicted class value using argmax
                predictedClass = np.argmax(prediction)
                
                # converting predicted label number to character / digit and store in array
                labelNumToChar = charArray[predictedClass]
                self.predictedClassesArray.append(labelNumToChar)

                # Check if the predicted class is the same as the target label and increase the number of correct predictions
                if (predictedClass == labelsArray[i]):
                    num_correct += 1
                
                num_samples += 1
                i += 1
            # Calculate accuracy
            self.selectImagesAccuracy = round((float(num_correct) / float(num_samples))*100,2)
            
            print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
    
    def predict(self,model,image,savePath):
        # The model predicts the character or digit that is drawn by the user and returns the predicted result
        # Paramaters: model class (ie: LeNet5() or SpinalNet()), a image of a drawn letter or digit made by user and filepath 
        # the saved trained model
        # modified from:
        # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-predict-new-samples-with-your-pytorch-model.md
         with torch.no_grad():

            # Loading the saved model
            trainedModel = model
            trainedModel.load_state_dict(torch.load(savePath))
            trainedModel.eval()

            # Generate prediction
            prediction = trainedModel(image)

            # Predicted class value using argmax
            predictedClass = np.argmax(prediction)

            # Get probability of the classification
            probabilities = F.softmax(prediction, dim=1)
            topProbability, topClass = probabilities.topk(1,dim=1)
            accuracy = topProbability.numpy()[0]
            self.drawingAccuracy = round(accuracy[0] *100,2)
            return predictedClass
