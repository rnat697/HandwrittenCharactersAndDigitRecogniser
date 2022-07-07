
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from pyparsing import null_debug_action
import torch

from scripts.imageViewerGUI import imageViewerGUI

from scripts.importingGUI import importingGUI
from scripts.predictSelect import predictSelect
from scripts.trainSetGUI import trainSetGUI
from scripts.TrainOrTestModel import TrainOrTestModel
from scripts.LeNet5 import LeNet5
from scripts.SpinalNet import SpinalNet
from scripts.trainSetGUI import trainSetGUI

from skimage.transform import resize
import skimage

        
class mainGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.display()
        self.connectToolBar()

        #initalise arrays.
        self.savedModelNames= []
        self.savedModelFilePaths= []
        self.savedModelClasses= []
        self.savedEpochNums = []

    def initUI(self):
        #main window size, title
        self.setWindowTitle('Handwritten Character and Digit Recogniser')
        self.setGeometry(300, 300, 720, 580)  # (x,x,horizontal,vertical)

        self.setMinimumHeight(200)
        self.setMinimumWidth(300)

        label = QLabel('Once you trained your model\nPlease select on MenuBar\nUpdate --> \nUpdate Model Selection',self)
        label.move(540,250)



        #menubar
        self.initMenuBar()  #menu bar creation 
        self.initButtons()  #predict clear, buttons
        self.predictBox()   #draw box
        self.selectModel()  #train,test buttons
        self.paintBox()

    def initMenuBar(self): #menu bar is below title but above window.
        
        # Menu bar
        menuList = QMenuBar(self)
        fileMenu = QMenu("&File", self)
        menuList.addMenu(fileMenu)

        #add additional menus
        viewMenu = menuList.addMenu("&View")
        updateMenu = menuList.addMenu("Update")

        #Action creation list
            #import
        self.importAction = QAction('Import',self)
        fileMenu.addAction(self.importAction)
            
            #exit
        self.exitAction = QAction(QIcon('#icon/png if needed'), 'Exit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        fileMenu.addAction(self.exitAction)
           
            # View train set
        self.ViewTrainAction = QAction('View Trainset', self)
        viewMenu.addAction(self.ViewTrainAction)
            
            # View test set
        self.ViewTestAction = QAction('View Testset',self)
        viewMenu.addAction(self.ViewTestAction)

            # update combobox
        self.updateComboAction = QAction('Update Model Selection',self)
        updateMenu.addAction(self.updateComboAction)
       
    def initButtons(self): #initalise button
        yaxis=520
        #predict and clear
        self.btnPredict = QPushButton('Predict', self)
        self.btnClear = QPushButton('Clear',self)
        # Create another button for pausing and unpausing   #800,580 max x,y
        self.btnPredict.move(150, yaxis)
        self.btnClear.move(300,yaxis)     # x,y positioning

        #Button links
        self.btnPredict.clicked.connect(self.predictAction)
        self.btnClear.clicked.connect(self.clearAction)

    def predictBox(self):
        #creates box that displays predicted result.
        xaxis=540

        result = QLabel('Drawing Prediction Result', self)
        result.move(xaxis,380) #800,580

        #image display
        self.imageResult= QLabel(self)
        self.imageResult.move(xaxis,400)
        self.imageResult.resize(150,100)
        self.imageResult.setStyleSheet("border: 3px solid black;")
        self.imageResult.setFont(QFont('Arial', 20))
        self.imageResult.setAlignment(Qt.AlignCenter)

        self.predictL = QLabel('Accuracy:',self)
        self.predictL.move(xaxis,520)
        self.predictL.resize(150,50)
        

    def editAccuracy(self,x):
        #edits label value and sets it accordingly
        label = "Accuracy: " + str(x) + " %"
        self.predictL.setText(label)

    def selectModel(self):
        # comobo Box for Model selection
        xaxis=540

        self.cb = QComboBox(self)
        self.cb.addItem('--Select Model--')
        self.cb.addItem('LeNet')
        self.cb.addItem('SpinalNet')  
        self.cb.move(xaxis,50)
        self.cb.resize(150,40)        #(width,height,)

        #connects
        self.cb.activated.connect(self.cbSelectedModel) 


        # Buttons for Training and Testing model
        self.btnTrain = QPushButton('Train',self)
        self.btnTest =  QPushButton('Test',self)
        self.btnTest.clicked.connect(self.testAction)
        self.btnTrain.clicked.connect(self.trainAction)
        self.btnTrain.move(xaxis,120)
        self.btnTrain.resize(150,30)
        self.btnTest.move(xaxis,170)
        self.btnTest.resize(150,30)
     
    
    def paintBox(self):
        #Drawing canvas for user to draw their letter/digit on
        self.label= QLabel('Draw shape',self)
        self.label.setStyleSheet("border: 1px solid black;")
        self.offset = -50
        self.canvasDimension=450

        self.label.resize(self.canvasDimension,self.canvasDimension)  #dimensions of background to draw
        self.pixMap = QPixmap(self.label.size())
        self.pixMap.fill(Qt.white)      #fills pixmap with white background

        #sets blank pixmap
        self.label.setPixmap(self.pixMap)
        self.label.move(50,50)
        
        #resets position of last mouse position
        self.last_x, self.last_y = None,None

    def connectToolBar(self):
        # Connects tool bar buttons to corresponding dialogs
        self.importAction.triggered.connect(self.importTrigger)
        self.exitAction.triggered.connect(qApp.quit)
        self.ViewTrainAction.triggered.connect(self.trainsetViewer)
        self.ViewTestAction.triggered.connect(self.testsetViewer)
        self.updateComboAction.triggered.connect(self.updateComboBox)

    def cbSelectedModel(self, index):
        if(index ==0):
            self.model = null_debug_action #Select model name tag
        elif(index == 1):
            self.model = LeNet5()
            self.modelName = "LeNet5"
        elif (index == 2):
            self.model = SpinalNet()
            self.modelName = "SpinalNet"
        else:
            # Goes through respective saved model arrays and gets the saved name, model type (ie: SpinalNet or LeNet5) and file path of
            # the trained model. These are used when user wants to predict an image or drawing
            self.savedName = self.savedModelNames[index-3]
            self.savedModel = self.savedModelClasses[index-3]
            self.savedFilePath = self.savedModelFilePaths[index-3]
            self.savedEpoch = self.savedEpochNums[index-3]
    
    # Access the saved model instance from trainSetGUI and save the fields of the SavedModel class to an array
    # This is to be able to access the associated fields when user has chosen a saved trained model in the comobo box
    # and use it for paramaters in predictResults, predictSelect and TrainOrTestModel classes (ie when the user wants to use a trained 
    # model for predicting)
    def getSavedModelinstance(self):
        self.savedModelNames.append(self.trainSelection.getSavedModel().getTrainedModelName())
        self.savedModelFilePaths.append(self.trainSelection.getSavedModel().getTrainedFilePath())
        self.savedModelClasses.append(self.trainSelection.getSavedModel().getModel())
        self.savedEpochNums.append(self.trainSelection.getSavedModel().getEpochNum())

    # Inserts the latest saved model in the array to the combobox
    def updateComboBox(self):
        self.getSavedModelinstance()
        lastIndex = len(self.savedModelNames)-1
        latestName = self.savedModelNames[lastIndex]
        self.cb.addItem(latestName)

    # Opens up the Prediction Image Selection window
    def testAction(self):
        testImages = self.importInstance.getTestsetArray()
        testLabels = self.importInstance.getTestLabelArray()
        self.showTestSelection = predictSelect(testImages,testLabels,self.model,self.savedFilePath,self.savedEpoch)
        
    
    
    def trainAction(self):
        self.trainSelection = trainSetGUI(self.model, self.modelName)
        
    def clearAction(self):
        #sets pixmap back to white background
        self.pixMap = QPixmap(self.label.size())
        self.pixMap.fill(Qt.white)
        self.label.setPixmap(self.pixMap)

    # ----------- Others-----------

    def importTrigger(self):
        #Opens new window from https://stackoverflow.com/questions/45688873/click-menu-and-open-new-window
        self.importInstance = importingGUI()

    def trainsetViewer(self):
        # Opens new window for trainset viewing
        trainImages = self.importInstance.getTrainsetArray()
        trainLabels = self.importInstance.getTrainLabelArray()
        self.showtrainSetView = imageViewerGUI(trainImages,trainLabels,"Train")
    
    def testsetViewer(self):
        # Opens new window for testset viewing.
        testImages = self.importInstance.getTestsetArray()
        testLabels = self.importInstance.getTestLabelArray()
        self.showtrainSetView = imageViewerGUI(testImages,testLabels,"Test")


    # Modified code from: https://www.youtube.com/watch?v=qEgyGyVA1ZQ
    
    def mouseMoveEvent(self, event):
        # Function is used to create the drawing path of the mouse

        if self.last_x is None: 
            #records position of x,y coord of mouse 
            self.last_x = event.x() 
            self.last_y = event.y() 
            return

        #opens painter to draw on label
        self.painter = QPainter(self.pixMap)
        self.painter.setPen(QPen(Qt.black,25, cap= Qt.RoundCap))    #number = pen thickness
        self.painter.drawLine(self.last_x + self.offset, self.last_y + self.offset, event.x()+ self.offset, event.y()+ self.offset)
        self.painter.end()
        self.label.setPixmap(self.pixMap)   #updates line drawn send to display on label
        self.update()

        self.last_x = event.x() 
        self.last_y = event.y() 


    def mouseReleaseEvent(self, event):
        self.last_x = None
        self.last_y = None
        self.label.setPixmap(self.pixMap)


    def display(self):
        #shows gui
        self.show()

   
    def predictAction(self):
        # self.pixMap has the drawn letter/digit image
        # Convert the Pixelmap to a numpy array
        drawnImage = self.pixMap
        imageArray = self.qpixmapToArray(drawnImage)

        #Clears pixelmap but saves images.
        self.pixMap.fill(Qt.white)
        self.label.setPixmap(self.pixMap)

        # Manipulate the image so that it has a similar format to EMNIST (see figure 1 in https://arxiv.org/pdf/1702.05373v1.pdf)
        image = self.manipulateImage(imageArray)
       
        # Send image to prediction
        self.toPredict = TrainOrTestModel(self.savedEpoch)
        
        # Get the predicted label result and accuracy
        predictLabelResult = self.toPredict.predict(self.savedModel,image,self.savedFilePath)
        predictAccuracy = self.toPredict.getDrawingAccuracy()
        
        self.editAccuracy(predictAccuracy)

        # converting label number to character 
        charArray = ["0","1","2","3","4","5","6","7","8","9",
                    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        predictedChar = charArray[predictLabelResult]
        self.imageResult.setText(predictedChar)
        
        
# ---------- Image Manipulation for drawn image to predict ------
    # Manipulate the image as seen in figure 1 in https://arxiv.org/pdf/1702.05373v1.pdf to look similar to EMNIST
    def manipulateImage(self, imageArray):
        image = np.array(imageArray)
        image = image.astype(np.float32)
        image = image[:,:,0] # Switch to single channel

        # Crop image using ROI and apply padding
        y_Min, y_Max, x_Min, x_Max = (0, 0, 0, 0) 
        y_Min = self.first_nonzero(image, axis = 0, invalid_val = -1) 
        y_Min = (y_Min[y_Min >= 0]).min() 
        x_Min = self.first_nonzero(image, axis = 1, invalid_val = -1) 
        x_Min = (x_Min[x_Min >= 0]).min() 
        y_Max = self.last_nonzero(image, axis = 0, invalid_val = -1) 
        y_Max = (y_Max[y_Max >= 0]).max() 
        x_Max = self.last_nonzero(image, axis = 1, invalid_val = -1) 
        x_Max = (x_Max[x_Max >= 0]).max() 
        cropped_np_im = image[y_Min:y_Max, x_Min:x_Max]
        image =  cropped_np_im

        # apply gaussian blur filter
        image = skimage.filters.gaussian(image, sigma=1)
        
        # Resize size the image to be a square
        currentSize= image.shape
        if (currentSize[0] > currentSize[1]):
            image = resize(image,(currentSize[0],currentSize[0]))
        else:
            image = resize(image,(currentSize[1],currentSize[1]))
        # pad image with padding size of 2 
        padded_image = np.pad(image,2)
        image = padded_image

        # Downscale the image to be 28x28
        # modified from https://chowdera.com/2021/12/202112211337132058.html part 5
        image = resize(image, (28, 28))# Convert to 28*28 Size 
        image = torch.from_numpy(image)# Convert to tensor 
        image = torch.unsqueeze(image, dim = 0)# Add a dimension 
        image = torch.unsqueeze(image, dim = 0)/255. # Add another dimension and map the gray scale to (between 0 and 1) 
        return image

    # Converting pixmap to a numpy Array for image manipulation
    def qpixmapToArray(self,pixmap):
        #Modified from: https://stackoverflow.com/questions/45020672/convert-pyqt5-qpixmap-to-numpy-ndarray

        # Scale down pixelmap to 128 x 128 as seen in https://arxiv.org/pdf/1702.05373v1.pdf figure 1
        pixmap = pixmap.scaled(128,128, Qt.KeepAspectRatio, Qt.SmoothTransformation) # scale it down to 128 x 128 for image manipulation
        
        img_size = pixmap.size()
        channels_count = 4
        image = pixmap.toImage()
        image.invertPixels() # invert colours so that character/digit is in white and background is black

        s = image.bits().asstring(img_size.width() * img_size.height() * channels_count)
        arr = np.fromstring(s, dtype=np.uint8).reshape((img_size.height(), img_size.width(), channels_count)) 
        return arr

   # --- Finding nonzero ROI --
   # To find the non-zero Region Of Interest of the image. Where a non-zero number (white) indicates the drawing and a zero indicates
   # the background (black).
   # Modified from https://stackoverflow.com/questions/66440022/get-non-zero-roi-from-numpy-array
    def first_nonzero(self,arr, axis, invalid_val=-1):
        mask = arr!=0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)-2 # 2 pixel padding
    def last_nonzero(self,arr, axis, invalid_val=-1):
        mask = arr!=0
        val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)+4 # 4 pixel padding

