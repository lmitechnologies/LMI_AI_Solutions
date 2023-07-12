from ctypes import *
import threading
from typing import AnyStr, Callable

kOK = 1
kNULL = 0

class MsgManager:
    #Initializes Mangager
    def __init__(self, GoSdk, system, dataset):
        self.GoSdk = GoSdk
        self.system = system
        self.dataset = dataset
    #Data handler with a variable timeout and callback function
    def SetDataHandler(self, timeout: int, function: Callable) -> None:
        if function != kNULL:
            self.thread = threading.Thread(target= self.Worker, args=[timeout, function])
            self.start = True
            self.thread.daemon = True #Used to make sure thread exits safely upon software exit
            self.thread.start()       
        else:
            self.start = False
            self.thread.join()
    #Worker thread that only waits for data and then passes to callback function
    def Worker(self, timeout, function):
        while self.start:
            if (self.GoSdk.GoSystem_ReceiveData(self.system, byref(self.dataset), timeout) == kOK):
                function(self.dataset)