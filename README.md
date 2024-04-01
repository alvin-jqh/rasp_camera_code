# Smart Shopping Cart, Raspberry Pi Source Code
3rd Year Project by Alvin Huang
Note: This requires an Arduino to be connected and running the code found at https://github.com/alvin-jqh/Arduino_code/tree/master   

# Setup
This code was tested using Python 3.11 and has been tested on windows and linux.    

To work on linux, 2 changes must be done:  
all the backward slashes ```\``` in any file paths must be replaced by a forward slash```/```  
line 18 of camera_class.py must be changed to:   ```self.cap = cv2.VideoCapture(self.camera_id)```  

Additional packages can be found in requirements.txt and can be installed using command
```
pip install -r requirements.txt
```

# Running the code
This requires two cameras to run the code
The camera IDs can be changed in the line where the main function is called


There are 2 pieces of code that can be run:  
``` double_new_method.py```  
This code only contains the image processing part of the algorithm and no control or communications.
Opens windows to show the camera feed and results of the image processing

```following_no_screen.py```  
This code contains all the necessary functionality for human following. Will only run if an Arduino and 2 cameras are connected.
The serial port and baud rate of the Arduino can be changed when calling the main function in line 170
