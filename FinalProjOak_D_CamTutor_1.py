#imports
import depthai as dai
import cv2
import numpy as np
import FinalProjOak_D_Func as funcs


if __name__ == '__main__':
    pipeline = dai.Pipeline() #creates pipeline

    #set up cameras
    monoLeft = funcs.getMonoCamera(pipeline, isLeft = True)
    monoRight = funcs.getMonoCamera(pipeline, isLeft = False)

    #create camera outputs
    xoutLeft = pipeline.createXLinkOut()
    xoutLeft.setStreamName('left')

    xoutRight = pipeline.createXLinkOut()
    xoutRight.setStreamName('right')

    #associate cameras with outputs
    monoLeft.out.link(xoutLeft.input)
    monoRight.out.link(xoutRight.input)

    #hook pipeline to camera
    with dai.Device(pipeline) as device:
        #gets output ques
        leftQueue = device.getOutputQueue(name = 'left', maxSize = 5)
        rightQueue = device.getOutputQueue(name = 'right', maxSize = 5)

        #config display window
        cv2.namedWindow('Stereo Pair')
        sideBySide = True #change to side by side or single frame

    while True:
        #get frames
        leftFrame = funcs.getFrame(leftQueue)
        rightFrame = funcs.getFrame(rightQueue)

        #sets side by side or overlap based on config settings
        if sideBySide:
            imOut = np.hstack((leftFrame, rightFrame))
        else:
            imOut = np.uint8(leftFrame/2 + rightFrame/2)
        
        #user commands
        key = cv2.waitKey(1)
        if key == ord('q'):
            break #quits window when q is pressed
        elif key == ord('t'):
            sideBySide = not sideBySide #toggles side by side when t is hit