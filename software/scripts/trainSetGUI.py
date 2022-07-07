from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

from scripts.trainProgressGUI import trainProgressGUI
from scripts.savedModel import SavedModel

class trainSetGUI(QWidget):
    # To load the train set and allow user to set trainset ratio, epoch number, batch number 
    # and a name for the trained model for training
    def __init__(self, model, modelName):
        super().__init__()
        self.model = model
        self.modelName  = modelName
        self.initUI()
        self.display()
        self.epochNumber = 0
        self.batchSize = 0
        self.modelName = ""
        self.trainRatio = 0


    
    def initUI(self):
        self.setWindowTitle('Load Training set')
        self.setGeometry(500,300,600, 350)  # (x on screen ,y on screen,horizontal,vertical)

        #title
        titleString = 'Train Dataset with ' + str(self.modelName) + ' Model'
        Title = QLabel(titleString, self)
        Title.setFont(QFont('Arial', 15))
        Title.setStyleSheet("border: 3px solid black;")
        Title.move(100,20)
    


        self.setupSlider()
        self.editLine()
        self.addBtns()

    def setupSlider(self):
        self.slider = QSlider(Qt.Horizontal, self)
        self.sliderY = 75

        self.slider.move(100,self.sliderY)    #x,y
        self.slider.resize(350,20)
        self.slider.setRange(0, 100)
        self.slider.setTickInterval(5) 
        self.slider.setTickPosition(3)      #state of the grooves
        self.slider.sliderMoved.connect(self.percentChange)

        self.percent = QLabel('0 %',self)
        self.percent.move(475,self.sliderY-10)   #(x,y)
        self.percent.setFont(QFont('Arial', 15))
        self.percent.resize(80,40)
        

    def percentChange(self):
        # Changes the text beside the slider to the current percentage value of the slider
        self.trainRatio = self.slider.value()
        self.percent.setText( str(self.trainRatio) + ' %' )


    def editLine(self):
        # intialising  and positioning editable boxes for user to input the batch size, epoch number and model name
        # and the text labels to indicate which text box is which
        batchSize = QLineEdit(self)
        epochNumber = QLineEdit(self)
        modelName = QLineEdit(self)
        
        batchSizeL =  QLabel('Batch Size :', self)
        epochNumberL = QLabel('Epoch Number :', self)
        modelNameL = QLabel('Model Name :', self)

        labelx= 125 + 25     #25 is the distance of label on x axis.
        labely= 125
        
        batchSize.move(labelx,labely)
        batchSize.textChanged.connect(self.batchTextChange)
        batchSizeL.move(25,labely)
        batchSize.resize(40,30)
        batchSizeL.setStyleSheet("border: 1px solid black;")

        epochNumber.move(labelx,labely+50)  #separate by 50 units
        epochNumber.textChanged.connect(self.epochTextChange)
        epochNumberL.move(25,labely+50)
        epochNumber.resize(40,30)
        epochNumberL.setStyleSheet("border: 1px solid black;")

        modelName.move(labelx,labely+100)   #increment
        modelName.textChanged.connect(self.nameTextChange)
        modelNameL.move(25,labely+100)
        modelName.resize(150,30)
        modelNameL.setStyleSheet("border: 1px solid black;")

    def addBtns(self):
        btnTrain = QPushButton('Train', self)
        btnCancel = QPushButton('Cancel',self)

        yaxis = 270

        btnTrain.move(100,yaxis)
        btnCancel.move(300,yaxis)

        btnTrain.clicked.connect(self.actionTrain)

    # -- On Textbox being edited --
    # Functions: epochTextChange, batchTextChange and nameTextChange
    # if the text has been changed in the text box we set the text to the associated variables
    def epochTextChange(self,number):
        self.epochNumber = number

    def batchTextChange(self,number):
        self.batchSize = number
    
    def nameTextChange(self,name):
        self.trainedName = name


    def actionTrain(self):
        # When Train button is pressed we want to open the training progress window
        if(self.epochNumber == 0 or self.batchSize == 0 or self.trainedName == "" or self.trainRatio == 0):
            # Blocking code so user doesn't train without these paramaters
            print("Please fill in all blanks")
        else:
            # Create a saved model instance
            filepath = "../software/"+ str(self.trainedName) + ".pt"
            print(filepath) # to check
            self.savedModel = SavedModel(self.model, self.epochNumber, self.trainedName,self.trainRatio,filepath)
            
            # open up progress bar window for training
            self.trainingProgress = trainProgressGUI(self.model,self.trainRatio,self.epochNumber,self.batchSize, self.trainedName)
            self.close()

    def actionCancel(self):
        self.close()

    # To return the saved model instance so we can access it in other classes for prediction of a drawing and prediction of selcted
    # images
    def getSavedModel(self):
        return self.savedModel

    def display(self):
        self.show()
