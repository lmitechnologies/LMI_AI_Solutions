##LMI internal Python script

from asyncio.windows_events import NULL
import os
import ctypes
from ctypes import *
from array import *
import csv
import numpy as np
from PIL import Image, ImageDraw
from gocators_utils.GoSdk_MsgHandler import MsgManager
import uuid
import cv2
import time
import pandas as pd
import open3d as o3d
from pcl_utils.point_cloud import PointCloud

### Load Api
# Please define your System Environment Variable as GO_SDK_4. It should reference the root directory of the SDK package.
SdkPath = os.environ['GO_SDK_4']
#Windows
kApi = ctypes.windll.LoadLibrary(os.path.join(SdkPath, 'bin', 'win64', 'kApi.dll'))
GoSdk = ctypes.windll.LoadLibrary(os.path.join(SdkPath, 'bin', 'win64', 'GoSdk.dll'))

#Linux
#kApi = ctypes.cdll.LoadLibrary(os.path.join(SdkPath, 'lib', 'linux_x64d', 'libkApi.dll'))
#GoSdk = ctypes.cdll.LoadLibrary(os.path.join(SdkPath, 'lib', 'linux_x64d', 'libGoSdk.dll'))

### Constant Declaration and Instantiation
kNULL = 0
kTRUE = 1
kFALSE = 0
kOK = 1
GO_DATA_MESSAGE_TYPE_SURFACE_POINT_CLOUD = 28
GO_DATA_MESSAGE_TYPE_MEASUREMENT = 10
GO_DATA_MESSAGE_TYPE_SURFACE_INTENSITY = 9
GO_DATA_MESSAGE_TYPE_UNIFORM_SURFACE = 8
GO_DATA_MESSAGE_TYPE_UNIFORM_PROFILE = 7
GO_DATA_MESSAGE_TYPE_PROFILE_POINT_CLOUD = 5
GO_DATA_MESSAGE_TYPE_STAMP = 0
RECEIVE_TIMEOUT = 10000

### Gocator DataType Declarations
kObject = ctypes.c_void_p
kValue = ctypes.c_uint32
kSize = ctypes.c_ulonglong
kAssembly = ctypes.c_void_p
GoSystem = ctypes.c_void_p
GoSensor = ctypes.c_void_p
GoDataSet = ctypes.c_void_p
GoDataMsg = ctypes.c_void_p
kChar = ctypes.c_byte
kBool = ctypes.c_bool
kCall = ctypes.c_bool
kCount = ctypes.c_uint32
k64f = ctypes.c_double
kStatus = ctypes.c_int32
k32s = ctypes.c_int32

class GoStampData(Structure):
    _fields_ = [("frameIndex", c_uint64), ("timestamp",c_uint64), ("encoder", c_int64), ("encoderAtZ", c_int64), ("status", c_uint64), ("id", c_uint32)]

class GoMeasurementData(Structure):
    _fields_ = [("numericVal", c_double), ("decision", c_uint8), ("decisionCode", c_uint8)]

class kIpAddress(Structure):
    _fields_ = [("kIpVersion", c_int32),("kByte",c_char*16)]

### Define Argtype and Restype
GoSdk.GoDataSet_At_argtypes = [kObject, kSize]
GoSdk.GoDataSet_At.restype = kObject
GoSdk.GoDataMsg_Type.argtypes = [kObject]
GoSdk.GoDataMsg_Type.restype = kValue
GoSdk.GoSurfaceMsg_RowAt.restype = c_int64
GoSdk.GoUniformSurfaceMsg_RowAt.restype = ctypes.POINTER(ctypes.c_int16)
GoSdk.GoSurfaceIntensityMsg_RowAt.restype = ctypes.POINTER(ctypes.c_uint8)
GoSdk.GoSurfacePointCloudMsg_RowAt.restype = ctypes.POINTER(ctypes.c_int16)
GoSdk.GoStampMsg_At.restype = ctypes.POINTER(GoStampData)
GoSdk.GoMeasurementMsg_At.restype = ctypes.POINTER(GoMeasurementData)
GoSdk.GoResampledProfileMsg_At.restype = ctypes.POINTER(ctypes.c_short)
GoSdk.GoSensor_CopyFile.argtypes = [GoSensor ,c_char_p, c_char_p]
GoSdk.GoSensor_FileNameAt.argtypes = [GoSensor,ctypes.c_uint64,c_char_p,ctypes.c_uint64]
GoSdk.GoProfileMsg_At.restype = ctypes.POINTER(ctypes.c_int16)
GoSdk.GoSensor_Transform.argtypes = [GoSensor]
GoSdk.GoSensor_Transform.restype = kObject
GoSdk.GoTransform_SetEncoderResolution.argtypes = [kObject, ctypes.c_double]
GoSdk.GoSensor_Setup.argtypes = [GoSensor]
GoSdk.GoSensor_Setup.restype = kObject
GoSdk.GoSetup_SetEncoderSpacing.argtypes = [kObject,k64f]
GoSdk.GoSetup_EncoderSpacing.argtypes= [kObject]
GoSdk.GoSetup_EncoderSpacing.restype = k64f
GoSdk.GoSetup_SetTriggerSource.argtypes = [kObject, k32s]
GoSdk.GoSetup_SetTriggerSource.restype = kStatus

def npytopcd(xyz_points, filename):
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)

    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud(filename+".pcd", pcd)
    return str(filename+".pcd")

def pcdtopng(filename, min, max,color_mapping):     
    pointcloud=PointCloud()
    pointcloud.read_points(filename,zmin=min,zmax=max,clip_mode=False)
    pointcloud.convert_points_to_image(color_mapping=color_mapping,contrast_enhancement=False,zmin_color=min,zmax_color=max)
    pointcloud.save_img(filename[:-4]+".png")
    return filename + ".png"

def getVersionStr():
    version = ctypes.create_string_buffer(32)
    myVersion = GoSdk.GoSdk_Version()
    kApi.kVersion_Format(myVersion, version, 32)
    return str(ctypes.string_at(version))

def kObject_Destroy(object):
    if (object != kNULL):
        kApi.xkObject_DestroyImpl(object, kFALSE)

def RecieveData(dataset):
    ## loop through all items in result message
    frameIndex = 0
    
    for i in range(GoSdk.GoDataSet_Count(dataset)):
        k_object_address = GoSdk.GoDataSet_At(dataset, i)
        dataObj = GoDataMsg(k_object_address)

        ## Retrieve stamp message
        if GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_STAMP:
            stampMsg = dataObj
            msgCount = GoSdk.GoStampMsg_Count(stampMsg)
            unique_filename = str(uuid.uuid4())
            for k in range(msgCount):
                stampDataPtr = GoSdk.GoStampMsg_At(stampMsg,k)
                stampData = stampDataPtr.contents
                print("frame index: ", stampData.frameIndex)
                print("time stamp: ", stampData.timestamp)
                print("encoder: ", stampData.encoder)
                print("sensor ID: ", stampData.id)
                frameIndex = stampData.frameIndex
                print()
        if GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_MEASUREMENT:
            measurementMsg = dataObj
            msgCount = GoSdk.GoMeasurementMsg_Count(measurementMsg)
            print("Measurement Message batch count: %d" % msgCount);

            for k in range(GoSdk.GoMeasurementMsg_Count(measurementMsg)):
                measurementDataPtr = (GoSdk.GoMeasurementMsg_At(measurementMsg, k))
                measurementData = measurementDataPtr.contents #(measurementDataPtr, POINTER(GoMeasurementData)).contents
                measurementID = (GoSdk.GoMeasurementMsg_Id(measurementMsg))
                print("Measurement ID: ", measurementID)
                print("Measurement Value: ", measurementData.numericVal)
                print("Measurment Decision: " + str(measurementData.decision))
                print()
        elif GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_UNIFORM_SURFACE:
            surfaceMsg = dataObj
            print("Surface Message")

            #resolutions and offsets (cast to mm)
            XResolution = float((GoSdk.GoUniformSurfaceMsg_XResolution(surfaceMsg)))/1000000.0
            YResolution = float((GoSdk.GoUniformSurfaceMsg_YResolution(surfaceMsg)))/1000000.0
            ZResolution = float((GoSdk.GoUniformSurfaceMsg_ZResolution(surfaceMsg)))/1000000.0
            XOffset = float((GoSdk.GoUniformSurfaceMsg_XOffset(surfaceMsg)))/1000.0
            YOffset = float((GoSdk.GoUniformSurfaceMsg_YOffset(surfaceMsg)))/1000.0
            ZOffset = float((GoSdk.GoUniformSurfaceMsg_ZOffset(surfaceMsg)))/1000.0
            width = GoSdk.GoUniformSurfaceMsg_Width(surfaceMsg)
            length = GoSdk.GoUniformSurfaceMsg_Length(surfaceMsg)
            size = width * length

            print("Surface data width: " + str(width))
            print("Surface data length: " + str(length))
            print("Total num points: " + str(size)) 

            #Generate Z points
            start = time.time()
            surfaceDataPtr = GoSdk.GoUniformSurfaceMsg_RowAt(surfaceMsg, 0)
            Z = np.ctypeslib.as_array(surfaceDataPtr, shape=(size,))  
            Z = Z.astype(np.double)
            #remove -32768 and replace with nan
            Z[Z==-32768] = np.nan    
            #scale to real world units                
            Z = (Z * ZResolution) + ZOffset     
            print("Z array generation time: ",time.time() - start)

            #generate X points
            start = time.time()
            X = (np.asarray(range(width), dtype=np.double) * XResolution) + XOffset
            X = np.tile(X, length)
            print("X array generation time: ",time.time() - start)

            #generate Y points
            start = time.time()
            Y = (np.arange(length, dtype=np.double)* YResolution) + YOffset
            Y = np.repeat(Y, repeats=width)
            print("Y array generation time: ",time.time() - start)

            #Generate X, Y, Z array for saving
            data_3DXYZ = np.stack((X,Y,Z), axis = 1)

            unique_filenameUniform = unique_filename + "_uniform"

            #Save NPY File
            npyNameOut = unique_filenameUniform + ".npy"
            np.save(npyNameOut, data_3DXYZ)

            #Save PCD File
            pcdNameOut = npytopcd(data_3DXYZ,unique_filenameUniform)
            print("saved file: " + pcdNameOut)

            #Save PNG File
            #Best practice is to set max and min colors (in mm) by hand
            if(np.any(Z<0)):
                print("[INFO]: Warning, some Z points are below 0. Points below 0 are scales to black (0,0,0)")
            pngNameOut = pcdtopng(pcdNameOut, 0, 0, "rainbow")
            print("saved file: " + pngNameOut)
            print()

        elif GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_SURFACE_INTENSITY:
            print("Intensity Message")
            surfaceIntensityMsg = dataObj

            #resolutions and offsets (cast to mm)
            XResolution = float((GoSdk.GoSurfaceIntensityMsg_XResolution(surfaceIntensityMsg)))/1000000.0
            YResolution = float((GoSdk.GoSurfaceIntensityMsg_YResolution(surfaceIntensityMsg)))/1000000.0
            XOffset = float((GoSdk.GoSurfaceIntensityMsg_XOffset(surfaceIntensityMsg)))/1000.0
            YOffset = float((GoSdk.GoSurfaceIntensityMsg_YOffset(surfaceIntensityMsg)))/1000.0
            width = GoSdk.GoSurfaceIntensityMsg_Width(surfaceIntensityMsg)
            length = GoSdk.GoSurfaceIntensityMsg_Length(surfaceIntensityMsg)
            size = width * length
            
            print("Surface data width: " + str(width))
            print("Surface data length: " + str(length))
            print("Total num points: " + str(size)) 

            #Generate I points
            surfaceIntensityDataPtr = GoSdk.GoSurfaceIntensityMsg_RowAt(surfaceIntensityMsg, 0)
            I = np.array((surfaceIntensityDataPtr[0:width*length]), dtype=np.uint8)
            
            #generate X points
            start = time.time()
            X = (np.asarray(range(width), dtype=np.double) * XResolution) + XOffset
            X = np.tile(X, length)
            print("X array generation time: ",time.time() - start)

            #generate Y points
            start = time.time()
            Y = (np.arange(length, dtype=np.double)* YResolution) + YOffset
            Y = np.repeat(Y, repeats=width)
            print("Y array generation time: ",time.time() - start)

            #Generate X, Y, Z array for saving
            data_3DXYI = np.stack((X,Y,I), axis = 1)
            unique_filename = str(uuid.uuid4())
            unique_filenameIntensity = unique_filename + "_intensity"           

            #Save NPY File
            npyNameOut = unique_filenameIntensity + ".npy"
            np.save(npyNameOut, data_3DXYI)
            print("saved file: " + npyNameOut)

            #Save PCD File
            pcdNameOut = unique_filenameIntensity
            pcdNameOut = npytopcd(data_3DXYI,pcdNameOut)
            print("saved file: " + pcdNameOut)

            #Save PNG File
            #max and min are 0-255 by default. Intensity is represented at a 1 byte uint.
            pngNameOut = pcdtopng(pcdNameOut, 0, 255, "gray")
            print("saved file: " + pngNameOut)
            print()

        elif GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_SURFACE_POINT_CLOUD:
            surfaceMsg = dataObj
            print("Non-uniform Surface Message")
            #resolutions and offsets (cast to mm)
            XResolution = float((GoSdk.GoSurfacePointCloudMsg_XResolution(surfaceMsg)))/1000000.0
            YResolution = float((GoSdk.GoSurfacePointCloudMsg_YResolution(surfaceMsg)))/1000000.0
            ZResolution = float((GoSdk.GoSurfacePointCloudMsg_ZResolution(surfaceMsg)))/1000000.0
            XOffset = float((GoSdk.GoSurfacePointCloudMsg_XOffset(surfaceMsg)))/1000.0
            YOffset = float((GoSdk.GoSurfacePointCloudMsg_YOffset(surfaceMsg)))/1000.0
            ZOffset = float((GoSdk.GoSurfacePointCloudMsg_ZOffset(surfaceMsg)))/1000.0
            width = GoSdk.GoSurfacePointCloudMsg_Width(surfaceMsg)
            length = GoSdk.GoSurfacePointCloudMsg_Length(surfaceMsg)
            size = width * length
            dataLength = size*3 #each data point has an X, Y, Z component

    
            print("Surface data width: " + str(width))
            print("Surface data length: " + str(length))
            print("Total num points (X,Y,Z): " + str(dataLength))

        
            #Generate Z points
            start = time.time()
            surfaceDataPtr = GoSdk.GoSurfacePointCloudMsg_RowAt(surfaceMsg, 0)
            XYZ = np.ctypeslib.as_array(surfaceDataPtr, shape=(dataLength,))  
            XYZ = XYZ.astype(np.double)
            #remove -32768 and replace with nan
            XYZ[XYZ==-32768] = np.nan    
            #break into X, Y, Z lists
            X = XYZ[0::3]
            Y = XYZ[1::3]
            Z = XYZ[2::3]
    
            #scale to real world units (for Z only)                  
            Z = (Z * ZResolution) + ZOffset    
            print("Z array generation time: ",time.time() - start)
    
            #generate X points
            start = time.time()
            X = (X * XResolution) + XOffset
            print("X array generation time: ",time.time() - start)
    
            #generate Y points
            start = time.time()
            Y = (Y* YResolution) + YOffset
            print("Y array generation time: ",time.time() - start)

    
            #Generate X, Y, Z array for saving
            data_3DXYZ = np.stack((X,Y,Z), axis = 1)

            unique_filenamePC = unique_filename + "_PC"
            np.save(unique_filenamePC+".npy",data_3DXYZ)
            print("saved file: "+ unique_filenamePC+".npy")

            #Save PCD File
            pcdNameOut = npytopcd(data_3DXYZ,unique_filenamePC)
            print("saved file: " + pcdNameOut)
            print()
        


    kObject_Destroy(dataset)



if __name__ == "__main__":
    # Instantiate system objects
    api = kAssembly(kNULL)
    system = GoSystem(kNULL)
    sensor = GoSensor(kNULL)
    dataset = GoDataSet(kNULL)
    dataObj = GoDataMsg(kNULL)
    changed = kBool(kNULL)

    print('Sdk Version is: ' + getVersionStr())

    GoSdk.GoSdk_Construct(byref(api))  # Build API
    GoSdk.GoSystem_Construct(byref(system), kNULL)  # Construct sensor system

    #connect to sensor via IP
    sensor_IP = b"127.0.0.1" #default for local emulator is 127.0.0.1
    ipAddr_ref = kIpAddress()
    kApi.kIpAddress_Parse(byref(ipAddr_ref), sensor_IP)
    GoSdk.GoSystem_FindSensorByIpAddress(system,byref(ipAddr_ref),byref(sensor))
    
    #connect to sensor via ID
    #sensor_ID = 54384
    #GoSdk.GoSystem_FindSensorById(system, sensor_ID, byref(sensor))

    GoSdk.GoSensor_Connect(sensor)  # Connect to the sensor
    GoSdk.GoSystem_EnableData(system, kTRUE)  # Enable the sensor's data channel to receive measurement data
    #GoSdk.GoSensor_Start(sensor)  # Start the sensor to gather data
    print("connected!")

    #Initialize message handler manager
    Mgr = MsgManager(GoSdk, system, dataset)

    #Set data handler which spawns a worker thread to recieve input data
    Mgr.SetDataHandler(RECEIVE_TIMEOUT, RecieveData)

    #Issue a stop then start incase the emulator is still running. For live sensors, only a start is needed.
    GoSdk.GoSensor_Stop(sensor) 
    #GoSdk.GoSensor_Start(sensor)
    
    #Do nothing
    while(input() != "exit"):
        pass
    
    #Can close thread manually by recalling data handler with kNull passed
    #Mgr.SetDataHandler(GoSdk, system, dataset, RECEIVE_TIMEOUT, kNULL)


    ### Destroy the system object and api
    kObject_Destroy(system)
    kObject_Destroy(api)
    print("Done")


