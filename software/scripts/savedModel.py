
# class to store saved models and associated values
# this should be used first before calling the test function in TrainOrTestModel.py
class SavedModel():
    def __init__(self,model,epochNum, name, trainPercent, filePath):
        # Parameters: the model class (ie: SpinalNet() or LeNet()), epoch number, name of saved model, training % (training ratio),
        # file path of saved model in format of ../software/filename.pt
        self.model = model
        self.trainedModelName =  name
        self.epochNum = epochNum
        self.trainPercent = trainPercent
        self.trainedFilePath = filePath
    
    def getModel(self):
        return self.model
    
    def getTrainedModelName(self):
        return self.trainedModelName
    
    def getEpochNum(self):
        return self.epochNum
    
    def getTrainPercent(self):
        return self.trainPercent
   
    def getTrainedFilePath(self):
        return self.trainedFilePath