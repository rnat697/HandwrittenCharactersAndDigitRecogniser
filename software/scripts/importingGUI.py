from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from pathlib import *
from scripts.Dataset import Dataset


class DownloadDataThread(QThread):
    # Thread to download EMNIST dataset
    finishButton = pyqtSignal()
    
    def run(self):
        self.data = Dataset()        
        self.data.downloadDataset()

        # Loads entire set of test and train images here because it takes a long time to load it all
        self.data.loadSetsAndConvertToNumpy()
        self.testArray = self.data.getTestsetArray()
        self.trainArray = self.data.getTrainsetArray()
        self.testLabel = self.data.getTestLabelArray()
        self.trainLabel = self.data.getTrainLabelArray()
        self.finishButton.emit();

    def returnTestsetArray(self):
        return self.testArray
    def returnTrainsetArray(self):
        return self.trainArray
    def returnTestsetLabels(self):
        return self.testLabel
    def returnTrainLabels(self):
        return self.trainLabel

    def stop(self):
        self.terminate()
        

    
            
    

class ProgressThread(QThread):

    updateProgress = pyqtSignal(int)
    updateTime = pyqtSignal(int)
    updateButton = pyqtSignal()
    
    
    def run(self):
        # Updates progress bar and time left depending on the size of the folder
        # The folder size is checked to see if it matches the byte size of EMNISt folder when its fully downloaded
        foldersize = 2332486384    #bytes size of EMNIST dataset folder
        pathEmnist = "..\\software\\emnist_data\\EMNIST\\raw"
        fileSize = sum(p.stat().st_size for p in Path(pathEmnist).rglob('*'))   

        # Updates progress bar on the GUI 
        while(fileSize != foldersize):
            fileSize = sum(p.stat().st_size for p in Path(pathEmnist).rglob('*'))
            n = (fileSize/foldersize)*100

            if (fileSize == 0):
                fileSize =1 #make sure that time needed does not divide by 0           

            timeNeeded = foldersize/fileSize
            self.updateProgress.emit(n)
            self.updateTime.emit(timeNeeded)
            
        self.updateProgress.emit(100) # if the import has already been downloaded before
        self.updateTime.emit(-1)




class importingGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.display()

    def initUI(self):
        self.setWindowTitle('Importing in progress')
        self.setGeometry(300, 300, 600, 200)  # (x,x,horizontal,vertical)

        self.setMinimumHeight(200)
        self.setMinimumWidth(300)        

        self.pbarSetup();
        self.controlImportButton();  

        self.timer = QBasicTimer()
        self.step = 0;
   
    def controlImportButton(self):
        # Start and stop buttons for importing
        # Push Button 
        self.btnStart = QPushButton('Start', self)
        self.btnStop = QPushButton('Stop',self)

        # Create another button for pausing and unpausing
        ybutton = 150              #600,200
        xbutton = 150
        interval = 150                #spacing
        self.btnStart.move(xbutton, ybutton)
        self.btnStart.clicked.connect(self.startAction)
    
        self.btnStop.move(xbutton+interval, ybutton)
        self.btnStop.clicked.connect(self.cancelAction)


    
    def pbarSetup(self):
        # Initialises progress bar
        Title = QLabel('Downloading EMNIST Dataset', self)
        self.timeTaken = QLabel('Time left x min x second',self)
        Title.move(200,10)  
        self.timeTaken.move(210,50)
        self.timeTaken.resize(400,20)
      
        
        #progressbar
        self.pbar = QProgressBar(self)
        self.pbar.move(50,100)         #600,200
        self.pbar.setMaximum(100)
        self.pbar.resize(500,30)           #width , height
        self.pbar.setValue(0)

    def updateProgressBar(self,n):
        # update progress bar value depending on the value emitted from Progress thread
        self.pbar.setValue(n)   

    def startAction(self):
        # Start button is clicked, download of dataset starts and progress bar updates via threads
        self.downloadSet = DownloadDataThread() 
        self.progress = ProgressThread()

        self.progress.updateProgress.connect(self.updateProgressBar)
        self.downloadSet.finishButton.connect(self.finishAction)
        self.progress.updateTime.connect(self.updateTimeLabel)

        self.downloadSet.start()
        self.progress.start()

        # waits till both processes are done
        # start button is disable during this process
        self.btnStart.setDisabled(True);
        self.btnStart.setText('Disabled')
    

    def cancelAction(self):
        # Stops download thread and start button is enabled if user wants to start download again (from the beginning)
        self.downloadSet.stop()
        self.pbar.setValue(0)
        self.btnStart.setEnabled(True)
        self.btnStart.setText('Start')

    def updateTimeLabel(self,time):
        # Updates time left of downloading
        x1 = int(time/60)
        x2 = int(time - (x1*60))

        timeTakenstring = 'Time left ' + str(x1) + ' min ' + str(x2) + ' second '

        if (x2==-1):
            timeTakenstring ='The download and unpacking has finished.'   
            
            self.timeTaken.move(150,50)


        self.timeTaken.setText(timeTakenstring)

       

    def finishAction(self):
        # Finished button appears on the "Start" button when download is finished
        # This will close the window
        self.btnStart.clicked.disconnect(self.startAction)
        self.btnStart.setText('Finish')
        self.btnStart.setEnabled(True)
        self.btnStart.clicked.connect(self.closeWindow)
    
    def closeWindow(self):
        self.close()

    def display(self):
        self.show()
    

# --- To pass this particular instance of Dataset to Image Viewer ---
# This is because we are loading and converting the train and test sets into numpy arrays in here as it would take a long time to load
# if we used the entire train and test set (697,932 images and 116,323 Images respectively)
    def getTrainsetArray(self):
        images = self.downloadSet.returnTrainsetArray()
        return images

    def getTrainLabelArray(self):
        labels = self.downloadSet.returnTrainLabels()
        return labels

    def getTestsetArray(self):
        images = self.downloadSet.returnTestsetArray()
        return images
    
    def getTestLabelArray(self):
        labels = self.downloadSet.returnTestsetLabels()
        return labels

