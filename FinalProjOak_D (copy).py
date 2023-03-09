#imports
import depthai as dai
import cv2
import numpy as np
import blobconverter
import FinalProjOak_D_Func as funcs
import time

#variables
frame_Size = (640, 360)
det_Input_Size = (300, 300)
model_name = 'face-detection-retail-0004'
zoo_type = 'depthai'
blob_path = None
frame_count = 0
fps = 0
prev_frame_time = 0
new_frame_time = 0
status_color = {'Face Detected': (0, 255, 0), 'No Face Detected': (0, 0, 255)}

if __name__ == '__main__':
    pipeline = dai.Pipeline() #creates pipeline

    #setup rgb camera
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(frame_Size[0], frame_Size[1])
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    #set up cameras
    monoLeft = funcs.getMonoCamera(pipeline, isLeft = True)
    monoRight = funcs.getMonoCamera(pipeline, isLeft = False)

    #Create Stereo
    stereo = pipeline.createStereoDepth()
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    #get blob
    if model_name is not None:
        blob_path = blobconverter.from_zoo(name = model_name, shaves = 6, zoo_type = zoo_type)

    #setup face detection
    face_spac_det_nn = pipeline.createMobileNetSpatialDetectionNetwork()
    face_spac_det_nn.setConfidenceThreshold(.75) #threshold for recognizing face
    face_spac_det_nn.setBlobPath(blob_path)
    face_spac_det_nn.setDepthLowerThreshold(100)
    face_spac_det_nn.setDepthUpperThreshold(5000)

    face_det_manip = pipeline.createImageManip()
    face_det_manip.initialConfig.setResize(det_Input_Size[0], det_Input_Size[1])
    face_det_manip.initialConfig.setKeepAspectRatio(False)

    #link RGP to ImageManip Nod
    cam.preview.link(face_det_manip.inputImage)
    face_det_manip.out.link(face_spac_det_nn.input)
    stereo.depth.link(face_spac_det_nn.inputDepth)

    #create stream
    x_preview_out = pipeline.createXLinkOut()
    x_preview_out.setStreamName('preview')
    cam.preview.link(x_preview_out.input)

    #get neural network output
    det_out = pipeline.createXLinkOut()
    det_out.setStreamName('det_out')
    face_spac_det_nn.out.link(det_out.input)

    #start pipeline
    with dai.Device(pipeline) as device:
        
        #output queues
        q_cam = device.getOutputQueue(name = 'preview', maxSize = 1, blocking = False)
        q_det = device.getOutputQueue(name = 'det_out', maxSize = 1, blocking = False)


        
        while True:
            #get right camera frame
            in_cam = q_cam.get()
            frame = in_cam.getCvFrame()

            bbox = None
            coordinates = None
            inDet = q_det.tryGet()

            if inDet is not None:
                detections = inDet.detections

                #if face detected
                if len(detections) != 0:
                    detection = detections[0]

                    #bounding box
                    xmin = max(0, detection.xmin)
                    ymin = max(0, detection.ymin)
                    xmax = min(detection.xmax, 1)
                    ymax = min(detection.ymax, 1)

                    #calc coordinates
                    x = int(xmin * frame_Size[0])
                    y = int(ymin * frame_Size[1])
                    w = int(xmax * frame_Size[0] - xmin * frame_Size[0])
                    h = int(ymax * frame_Size[1] - ymin * frame_Size[1])

                    bbox = (x, y, w, h)

                    #get special coordinates
                    coord_x = detection.spatialCoordinates.x
                    coord_y = detection.spatialCoordinates.y
                    coord_z = detection.spatialCoordinates.z
                    coordinates = (coord_x, coord_y, coord_z)
            #detects if face detected
            if bbox:
                status = 'Face Detected'
            else:
                status = 'No Face Detected'

            #display info
            funcs.display_info(frame, bbox, coordinates, status, status_color, fps)

            #calc avg fps
            if frame_count % 10 == 0:
                #time when we finish processing last 100 frames
                new_frame_time = time.time()

                #fps will be number of rame processed in a sec
                fps = 1 / ((new_frame_time - prev_frame_time) / 10)
                prev_frame_time = new_frame_time

            #keyboard commands
            key_pressed = cv2.waitKey(1) & 0xff
            if key_pressed == 27: #esc
                break

            #display final frame
            cv2.imshow('Face Cam', frame)

            #increment frame_count
            frame_count += 1

    cv2.destroyAllWindows()


