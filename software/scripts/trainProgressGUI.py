from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from scripts.Dataset import Dataset
from scripts.TrainOrTestModel import TrainOrTestModel
import time


class TrainingThread(QThread):
    # Thread to train the model
    def __init__(self, model,trainratio, epochNum, batchsize, name):
        super().__init__()
        self.modelClass = model
        self.trainRatio = int(trainratio)
        self.epochNumber = int(epochNum)
        self.batchSize = int(batchsize)
        self.trainedName = name
        self.training = TrainOrTestModel(self.epochNumber)

    
    def run(self):
        data = Dataset()
        data.downloadDataset()
        data.setBatchSize(self.batchSize)
        data.splitTrainAndValid(self.trainRatio)
        train = data.getTrainsetSplit()
        validate = data.getValidatesetSplit()
        print("training")
        self.training.training(self.modelClass,self.trainedName,train,validate)

    def getTraining(self):
        return self.training

    # modified code from https://stackoverflow.com/questions/26833093/how-to-terminate-qthread-in-python
    def stop(self):
        self.terminate()

class ProgressThread(QThread):
    # Thread to update progress bar depending on training progress / validation progress
    updateText = pyqtSignal(int)
    updateProgress = pyqtSignal(int)
    updateAccuracy = pyqtSignal(float)
    updateLoss = pyqtSignal(float)
    updateEpoch = pyqtSignal(int)
    finishButton = pyqtSignal()
    def __init__(self,trainInstance):
        super().__init__()
        self.train = trainInstance
    
    def run(self):
        currentEpoch = self.train.getCurrentEpoch()
        maxEpoch = self.train.getMaxEpoch()
        trainProgress = self.train.getTrainingProgress()
        self.updateText.emit(1)
        self.updateEpoch.emit(currentEpoch+1)
        prevEpoch = currentEpoch

        #--- Tracking Training Progress ---
        # Check training progress while the current epoch number is less than the maximum epoch
        while(currentEpoch < maxEpoch):

            # Checking train progress
            time.sleep(1)
            trainProgress = self.train.getTrainingProgress()

            currentEpoch = self.train.getCurrentEpoch()
            if (prevEpoch < currentEpoch): # checks if epoch number has changed and emits the epoch iteration number
                prevEpoch = currentEpoch
                self.updateEpoch.emit(currentEpoch+1)
            
            # stops loop when current epoch number has reached max and training progress is 100 and moves on to validation
            if(currentEpoch == maxEpoch-1 and trainProgress >= 99): 
                self.updateProgress.emit(trainProgress)
                break
            # Emit to progess bar
            self.updateProgress.emit(trainProgress)

        #--- Tracking validating Progress ---
        self.updateText.emit(2)
        validateProgress = self.train.getValidateProgress()
        self.updateText.emit(3)
        while(validateProgress <=100):

            # Check validation progress
            time.sleep(0.5)
            validateProgress = self.train.getValidateProgress()

            # stop tracking validation progress when it reaches 100
            if(validateProgress == 100):
                self.updateProgress.emit(validateProgress)
                break
            # Emit to progess bar
            self.updateProgress.emit(validateProgress)
        
        #--- Displaying stats and confirmation of finished training ----
        self.updateText.emit(4)
        accuracy = self.train.getTrainAccuracy()
        aveloss = self.train.getAverageLoss()
        self.updateAccuracy.emit(accuracy)
        self.updateLoss.emit(aveloss)
        self.finishButton.emit()
    
    def stop(self):
        # modified code from https://stackoverflow.com/questions/26833093/how-to-terminate-qthread-in-python
        self.terminate()



class trainProgressGUI(QWidget):
    def __init__(self, model,trainratio, epochNum, batchsize, name):
        super().__init__()
        self.initUI()
        self.display()
        self.model = model
        self.trainRatio = trainratio
        self.epochNumber = epochNum
        self.batchSize = batchsize
        self.trainedName = name
        self.finished = False

    def initUI(self):
        self.setWindowTitle("Training In Progress")
        self.setGeometry(300, 300, 600, 300)  # (x,x,horizontal,vertical)

        self.setMinimumHeight(200)
        self.setMinimumWidth(300)        

        self.progressBar()
        self.terminalBox()

    def checkFinished(self):
        # Checks if validation progress has completely finished so we can update the text of start button to "Finish"
        return self.finished
   
    def terminalBox(self):
        # Text box to relay messages to user
        self.tb = QTextBrowser(self)
        self.tb.setAcceptRichText(True)
        self.tb.setOpenExternalLinks(True)
        self.tb.resize(500,150)
        self.tb.move(50,20)
        self.tb.append('Press the start button to begin training')

    def progressBar(self):
        # Initialising progress bar layout
        self.pbar = QProgressBar(self)
        self.pbar.move(50,200)         #600,300
        self.pbar.setMaximum(100)
        self.pbar.resize(550,30)           #width , height
        self.pbar.setValue(0)

        xaxis=150
        yaxis=250

        self.btnStart = QPushButton('Start', self)
        self.btnStart.move(xaxis, yaxis)
        self.btnStart.clicked.connect(self.startAction)

        self.btnStop = QPushButton('Cancel', self)
        self.btnStop.clicked.connect(self.cancelAction)
        self.btnStop.move(xaxis+200, yaxis)
    
    def startAction(self):
        # To start the training and progress threads
        self.trainingModel = TrainingThread(self.model,self.trainRatio,self.epochNumber,self.batchSize,self.trainedName)
        self.trainingModel.start()
        self.train = self.trainingModel.getTraining()

        self.progress = ProgressThread(self.train)    
        self.progress.updateProgress.connect(self.updateProgressBar)
        self.progress.updateEpoch.connect(self.updateEpochIterText)
        self.progress.updateText.connect(self.updateBox)
        self.progress.updateAccuracy.connect(self.updateAccuracyText)
        self.progress.updateLoss.connect(self.updateLossText)
        self.progress.finishButton.connect(self.finishAction)
        self.progress.start()

        # Disable the start button while training/validating in progress
        self.btnStart.setDisabled(True)
        self.btnStart.clicked.disconnect(self.startAction)

    def cancelAction(self):
        # Stops the threads and closes the window
        self.trainingModel.stop()
        self.progress.stop()
        self.close()

    def updateAccuracyText(self,percentage):
        # Updates accuracy text on the text box once validating has finished
        textAccuracy='Accuracy ' + str(percentage) + '%'
        self.tb.append(textAccuracy) 

    def updateLossText(self,aveLoss):
        # updates average loss text on the text box once validating has finished
        textAverageLoss='Average lost: '+ str(aveLoss)
        self.tb.append(textAverageLoss)

    def updateEpochIterText(self, epochNum):
        # updates epoch text on text box on epoch number increase
        # Since during training progress bar goes from 0 to 100% every epoch iteration
        textIteration = "Training Epoch: " + str(epochNum)
        self.tb.append(textIteration)

    def updateBox(self,signal):
        # update text in text box depending on signal recieved from progress thread
        if (signal == 1):
            textTrain = 'Training Model...'
            self.tb.append(textTrain) 

        if (signal == 2):
            textTrainComplete = 'Training Model Complete'
            self.tb.append(textTrainComplete) 

        if (signal == 3):
            textValidate='Validating...'
            self.tb.append(textValidate) 

        if (signal == 4):
            textValidate='Validating Complete'
            self.tb.append(textValidate) 


    def updateProgressBar(self,n):
        self.pbar.setValue(n)   

    def finishAction(self):
        # once validation progress is finished, finish button appears and when clicked it will close the window
        self.btnStart.setText('Finish')
        self.btnStart.setEnabled(True)
        self.btnStart.clicked.connect(self.closeWindow)

    def closeWindow(self):
        self.close()

    def display(self):
        self.show()