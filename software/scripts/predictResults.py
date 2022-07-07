
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import sys
from scripts.TrainOrTestModel import TrainOrTestModel

class PredictingThread(QThread):
    # Thread to predict selected images
    updatePredictionsHover = pyqtSignal(object)
    updateAccuracy = pyqtSignal(float)
    updateText = pyqtSignal(int)

    def __init__(self, model, epochNum,filePath, images, labels):
        super().__init__()
        #parameters to run the DNN
        self.modelClass = model
        self.savedFilePath = filePath
        self.epochNumber = epochNum
        self.images = images
        self.targetLabels = labels

        

    def run(self):
        self.updateText.emit(1) # to show predictions are being run to user

        test  = TrainOrTestModel(self.epochNumber)
        test.testing(self.modelClass, self.images, self.targetLabels,self.savedFilePath)
        predictionsList = test.getPredictedClasses() # on hover user can see the predicted class
        accuracy = test.getSelectedImagesAccuracy()
        self.updatePredictionsHover.emit(predictionsList)   #sends to predictionlist function in GUi
        self.updateAccuracy.emit(accuracy)          #updates accuracy in Gui

        # Update text so user knows that predictions has finished
        self.updateText.emit(2)
        self.updateText.emit(3)

class predictResults(QWidget):
    def __init__(self,images,labels, model, filepath,epoch):

        super().__init__()
        self.imgArray = []
        self.modelClass = model
        self.savedFilePath = filepath
        self.epochNum = epoch
        self.selectedImages = images
        self.selectedLabels =labels
        self.TotalNumOfImgs = len(self.selectedImages)
        self.initUI()
        self.showAllImages()
        self.display()
        

    def initUI(self):
        #sets title and loads GUI
        self.setWindowTitle('Prediction Results')
        self.setGeometry(300, 300, 800, 600)  # (x,x,horizontal,vertical)
        self.setMinimumHeight(200)
        self.setMinimumWidth(300)
        self.initialiseImages()
        self.createGrid()
        self.connectScrolltoGrid()
        self.predictionButtons()
        self.terminalBox()
        self.predictAccuracy()

    def predictionButtons(self):
       #loads buttons on GUI and connects them to corresponding functions
        self.btnStart = QPushButton('Start Prediction', self)
        self.btnClose = QPushButton('Exit',self)

        xaxis,yaxis= 590,270
        self.btnStart.move(xaxis,yaxis)
        self.btnStart.resize(200,40)
        self.btnStart.clicked.connect(self.runPrediction)
        self.btnClose.move(xaxis,yaxis+70)
        self.btnClose.resize(200,40)
        self.btnClose.clicked.connect(self.closeAction)
    
    
    def terminalBox(self):
    # Terminal box is to show prediction progress
        self.tb = QTextBrowser(self)
        self.tb.setAcceptRichText(True)
        self.tb.setOpenExternalLinks(True)
        self.tb.resize(200,150)
        self.tb.move(590,100)
        self.tb.append('Press the start button to begin predictions')
    

    def runPrediction(self):
        #triggers the thread runs parallel with gui
        self.runPredict = PredictingThread(self.modelClass,self.epochNum,self.savedFilePath,self.selectedImages,self.selectedLabels)
        self.runPredict.updateText.connect(self.updateBox)
        self.runPredict.updatePredictionsHover.connect(self.addOnHoverPredictions)
        self.runPredict.updateAccuracy.connect(self.editAccuracy)

        self.runPredict.run()
    
    def updateBox(self,signal):
        #udpates terminal accordingt to signal
        if (signal == 1):
            textTest = 'Predicting Images...'
            self.tb.append(textTest) 

        if (signal == 2):
            textTestComplete = 'Prediction Complete'
            self.tb.append(textTestComplete) 

        if (signal == 3):
            textInfo='Hover over images to see predicted values'
            self.tb.append(textInfo) 

    def closeAction(self):
        #close window
        self.close()
    
    def predictAccuracy(self):
        #sets up accuracy label
        self.predictL = QLabel('Accuracy:',self)
        self.predictL.move(650,450)
        self.predictL.resize(150,50)

    def editAccuracy(self,percentage):
        #update accuracy label
        accuracyText = "Accuracy: " + str(percentage) +"%"
        self.predictL.setText(accuracyText)
    
    def addOnHoverPredictions(self, predictions):
        # Adding on hover prediction value
        for i in range (self.TotalNumOfImgs):
            text = "Prediction is " + str(predictions[i])
            self.gridLayout.itemAt(i).widget().setToolTip(text)

    
    def initialiseImages(self):
    # Initialises images to a pixmap to show on GUI    
        for i in range (len(self.selectedImages)):
            image = self.selectedImages[i][0]
            height, width = np.shape(image)

            # Creating a pixel map of the image for display
            qImg = QImage(image, width, height, QImage.Format_Grayscale8)
            pixmap01 = QPixmap.fromImage(qImg).scaled(112,112, Qt.KeepAspectRatio) # scaling image so we can see it better on GUI
            
            self.imgArray.append(pixmap01)


    def showAllImages(self):
        #creates grid and loads images.
        # Displays all images

        for i in range (len(self.selectedImages)):
            self.addImage(i,i)

    
    def addImage(self,index,imgNum):
        # add the pixmap to a label on the GUI
    
        imgLabel = QLabel(self)
        imgLabel.setPixmap(self.imgArray[index])

        #Sets grid size row and column depending on pref
        if (imgNum%4==0):    # 4 images per row.
            self.column = 1
            self.row += 1

        else:
            self.column+=1

        #adds widget to gridlayout
        self.gridLayout.addWidget(imgLabel,self.row,self.column)



    def createGrid(self):
        #creates grid
        self.gridLayout = QGridLayout(self)
        self.column=1       #needed for inserting images.
        self.row=1          
    

    def connectScrolltoGrid(self):
        self.widget = QWidget()       # Scroll Area which contains the widgets, set as the centralWidget
        self.scroll = QScrollArea(self)

        self.widget.setLayout(self.gridLayout)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll.move(40,50)
        self.scroll.resize(550,500) #800 ,600


    def display(self):
        self.show()

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = predictResults()
   ex.display()
   sys.exit(app.exec_())