import depthai as dai
import cv2
import numpy as np

#gets queue transfers it to host, and converts to numpy array
def getFrame(queue):
    frame = queue.get() #get frame from queue
    return frame.getCvFrame() #converts to OpenCV format and returns

#sets up mono camera in pipeline
def getMonoCamera(pipeline, isLeft):
    # mono = pipeline.createMonoCamera() #configures mono camera
    mono = pipeline.create(dai.node.MonoCamera)
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P) #sets cameras resolution
    #checks if it is left or right camera
    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

#displays info on image frame
def display_info(frame, bbox, coordinates, status, status_color, fps):
    #display bounding box
    cv2.rectangle(frame, bbox, status_color[status], 2)

    #display coordinates
    if coordinates is not None:
        coord_x, coord_y, coord_z = coordinates
        cv2.putText(frame, f"Z: {int(coord_z)} mm", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)

    #create detail background
    cv2.rectangle(frame, (5, 5, 175, 100), (50, 0, 0), -1)

    #display authentication
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, status_color[status])
