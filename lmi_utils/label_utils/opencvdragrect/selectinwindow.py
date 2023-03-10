# MIT License

# Copyright (c) 2016 Akshay Chavan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import numpy as np


class Rect:
    '''
    Description: rectangle class captures x,y,w,h for all rectangle instances
    '''
    x = None
    y = None
    w = None
    h = None

    def printit(self):
        print(f'{str(self.x)}, {str(self.y)}, {str(self.w)}, {str(self.h)}')


# endclass

class dragRect:
    '''
    Description: Defines the draggable rectangle
    Objects for:
        -keepWithin: valid range (in the canvas)
        -outRect: output rectangle
        -anchor: stores distance from anchor point to top-left corner and bottom-right corner
    Normal Vars for:
        -sBlk: drag handle marker size
        -image: image rendered on canvas
        -wname: window name 
    State Vars for:
        -initialized 
        -return flag
        -active: True if rectangle already present
        -drag: Currently resizing rectangle
        -hold: currently holding mouse button
        -drag handles: True if currently pulling on a particular drag handle
    '''

    # Limits on the canvas
    keepWithin = Rect()
    # To store rectangle
    outRect = Rect()
    # To store rectangle anchor point
    # Here the rect class object is used to store
    # the distance in the x and y direction from
    # the anchor point to the top-left and the bottom-right corner
    anchor = Rect()
    # Selection marker size
    sBlk = 4
    # Whether initialized or not
    initialized = False

    # Image
    image_in = None
    image_runtime=None

    # Window Name
    wname = ""

    # Return flag
    returnflag = False

    # FLAGS
    # Rect already present
    active = False
    # Drag for rect resize in progress
    drag = False
    # Marker flags by positions
    TL = False
    TM = False
    TR = False
    LM = False
    RM = False
    BL = False
    BM = False
    BR = False
    # holding mouse button
    hold = False


# endclass

def init(dragObj, Img, windowName, windowWidth, windowHeight,x=0,y=0,w=0,h=0):
    '''
    Description: initializes the dragRect object dragObj
    
    Args: 
        -dragObj: user defined rectangle
    '''
    # Image
    dragObj.image_in = Img
    dragObj.image_runtime=Img

    # Window name
    dragObj.wname = windowName

    # Limit the selection box to the canvas
    dragObj.keepWithin.x = 0
    dragObj.keepWithin.y = 0
    dragObj.keepWithin.w = windowWidth
    dragObj.keepWithin.h = windowHeight

    # Set output rect to zero width and height
    dragObj.outRect.x = x
    dragObj.outRect.y = y
    dragObj.outRect.w = w
    dragObj.outRect.h = h

    if x==0 and y==0 and w==0 and h==0:
        pass
    else:
        dragObj.active=True
        clearCanvasNDraw(dragObj)

# enddef

def dragrect(event, x, y, flags, dragObj):
    '''
    Description: callback definition that is passed to cv2.setMouseCallback()

    Entrypoint to mouse behavior, click or movement.
    '''
    #-------------
    # Check if mouse is inside keepWithin rectangle
    # if no, set x,y to limit
    if x < dragObj.keepWithin.x:
        x = dragObj.keepWithin.x
    # endif
    if y < dragObj.keepWithin.y:
        y = dragObj.keepWithin.y
    # endif
    if x > (dragObj.keepWithin.x + dragObj.keepWithin.w - 1):
        x = dragObj.keepWithin.x + dragObj.keepWithin.w - 1
    # endif
    if y > (dragObj.keepWithin.y + dragObj.keepWithin.h - 1):
        y = dragObj.keepWithin.y + dragObj.keepWithin.h - 1
    # endif

    #-------------
    # switch on mouse action
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseDown(x, y, dragObj)
    # endif
    if event == cv2.EVENT_LBUTTONUP:
        mouseUp(x, y, dragObj)
    # endif
    if event == cv2.EVENT_MOUSEMOVE:
        mouseMove(x, y, dragObj)
    # endif
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseDoubleClick(x, y, dragObj)
    # endif

# enddef

def pointInRect(pX, pY, rX, rY, rW, rH):
    '''
    Description: check to see if mouse point inside of rectangle
    
    Args:
        pX: mouse point x
        pY: mouse point y
        rX: rect upper left x
        rY: rect upper left y
        rW: rect width
        rH: rect height
    '''
    if rX <= pX <= (rX + rW) and rY <= pY <= (rY + rH):
        return True
    else:
        return False
    # endelseif


# enddef

def mouseDoubleClick(eX, eY, dragObj):
    '''
    Description: double click event handler

    Args:
        -eX: current mouse pointer x
        -eY: current mouse pointer y
        -dragObj user defined rectangle 

    '''
    if dragObj.active:

        if pointInRect(eX, eY, dragObj.outRect.x, dragObj.outRect.y, dragObj.outRect.w, dragObj.outRect.h):
            dragObj.active=False
            dragObj.returnflag = True
            #cv2.destroyWindow(dragObj.wname)
        # endif

    # endif


# enddef

def mouseDown(eX, eY, dragObj):
    '''
    Description: 
        -main drag/resize callback
        -checks if user defined rectangle is active
            -yes:
                -checks for mouse pointer on drag handles
                -checks for mouse pointer on user defined rectangle.
                    -yes: then re-position
            -no:
                -create new user defined rectangle
                -initialize top left corner to mouse position 
                -sets drag to True for initial size setting
                -sets active to True for intial rect definition
    '''

    if dragObj.active:

        if pointInRect(eX, eY, dragObj.outRect.x - dragObj.sBlk,
                       dragObj.outRect.y - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.TL = True
            return
        # endif
        if pointInRect(eX, eY, dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk,
                       dragObj.outRect.y - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.TR = True
            return
        # endif
        if pointInRect(eX, eY, dragObj.outRect.x - dragObj.sBlk,
                       dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.BL = True
            return
        # endif
        if pointInRect(eX, eY, dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk,
                       dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.BR = True
            return
        # endif

        if pointInRect(eX, eY, dragObj.outRect.x + dragObj.outRect.w / 2 - dragObj.sBlk,
                       dragObj.outRect.y - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.TM = True
            return
        # endif
        if pointInRect(eX, eY, dragObj.outRect.x + dragObj.outRect.w / 2 - dragObj.sBlk,
                       dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.BM = True
            return
        # endif
        if pointInRect(eX, eY, dragObj.outRect.x - dragObj.sBlk,
                       dragObj.outRect.y + dragObj.outRect.h / 2 - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.LM = True
            return
        # endif
        if pointInRect(eX, eY, dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk,
                       dragObj.outRect.y + dragObj.outRect.h / 2 - dragObj.sBlk,
                       dragObj.sBlk * 2, dragObj.sBlk * 2):
            dragObj.RM = True
            return
        # endif

        # This has to be below all of the other conditions
        if pointInRect(eX, eY, dragObj.outRect.x, dragObj.outRect.y, dragObj.outRect.w, dragObj.outRect.h):
            dragObj.anchor.x = eX - dragObj.outRect.x
            dragObj.anchor.w = dragObj.outRect.w - dragObj.anchor.x
            dragObj.anchor.y = eY - dragObj.outRect.y
            dragObj.anchor.h = dragObj.outRect.h - dragObj.anchor.y
            dragObj.hold = True

            return
        # endif

    else:
        # dragObj.outRect.x = eX
        # dragObj.outRect.y = eY
        dragObj.drag = True
        dragObj.active = True
        return

    # endelseif


# enddef

def mouseMove(eX, eY, dragObj):
    '''
    Description: 
        -check for drag and active
            -if true: 
                -reset w and h using delta between mouse pointer and uL corner
                -update the image with the new rectangle
                -redraw rectangle
        -check if holding mouse button over main rectangle
            -if true:
                -move the rectangle
                -prevent movement outside the canvas
                -redraw rectangle
        -check if holding mouse button over drag handles
            -if true:
                -resize rectangle
                -redraw rectangle


    '''

    if dragObj.drag & dragObj.active:
        dragObj.outRect.w = eX - dragObj.outRect.x
        dragObj.outRect.h = eY - dragObj.outRect.y
        clearCanvasNDraw(dragObj)
        return
    # endif

    if dragObj.hold:
        # change rect position
        dragObj.outRect.x = eX - dragObj.anchor.x
        dragObj.outRect.y = eY - dragObj.anchor.y

        # prevent position outside canvas
        if dragObj.outRect.x < dragObj.keepWithin.x:
            dragObj.outRect.x = dragObj.keepWithin.x
        # endif
        if dragObj.outRect.y < dragObj.keepWithin.y:
            dragObj.outRect.y = dragObj.keepWithin.y
        # endif
        if (dragObj.outRect.x + dragObj.outRect.w) > (dragObj.keepWithin.x + dragObj.keepWithin.w - 1):
            dragObj.outRect.x = dragObj.keepWithin.x + dragObj.keepWithin.w - 1 - dragObj.outRect.w
        # endif
        if (dragObj.outRect.y + dragObj.outRect.h) > (dragObj.keepWithin.y + dragObj.keepWithin.h - 1):
            dragObj.outRect.y = dragObj.keepWithin.y + dragObj.keepWithin.h - 1 - dragObj.outRect.h
        # endif

        clearCanvasNDraw(dragObj)
        return
    # endif


    if dragObj.TL:
        dragObj.outRect.w = (dragObj.outRect.x + dragObj.outRect.w) - eX
        dragObj.outRect.h = (dragObj.outRect.y + dragObj.outRect.h) - eY
        dragObj.outRect.x = eX
        dragObj.outRect.y = eY
        clearCanvasNDraw(dragObj)
        return
    # endif
    if dragObj.BR:
        dragObj.outRect.w = eX - dragObj.outRect.x
        dragObj.outRect.h = eY - dragObj.outRect.y
        clearCanvasNDraw(dragObj)
        return
    # endif
    if dragObj.TR:
        dragObj.outRect.h = (dragObj.outRect.y + dragObj.outRect.h) - eY
        dragObj.outRect.y = eY
        dragObj.outRect.w = eX - dragObj.outRect.x
        clearCanvasNDraw(dragObj)
        return
    # endif
    if dragObj.BL:
        dragObj.outRect.w = (dragObj.outRect.x + dragObj.outRect.w) - eX
        dragObj.outRect.x = eX
        dragObj.outRect.h = eY - dragObj.outRect.y
        clearCanvasNDraw(dragObj)
        return
    # endif

    if dragObj.TM:
        dragObj.outRect.h = (dragObj.outRect.y + dragObj.outRect.h) - eY
        dragObj.outRect.y = eY
        clearCanvasNDraw(dragObj)
        return
    # endif
    if dragObj.BM:
        dragObj.outRect.h = eY - dragObj.outRect.y
        clearCanvasNDraw(dragObj)
        return
    # endif
    if dragObj.LM:
        dragObj.outRect.w = (dragObj.outRect.x + dragObj.outRect.w) - eX
        dragObj.outRect.x = eX
        clearCanvasNDraw(dragObj)
        return
    # endif
    if dragObj.RM:
        dragObj.outRect.w = eX - dragObj.outRect.x
        clearCanvasNDraw(dragObj)
        return
    # endif


# enddef

def mouseUp(eX, eY, dragObj):
    '''
    Description: action for releasing mouse button
        -disable drag state vars
        -disable resize state vars
        -check for 0 size:
            -if True, then deactivate "already present" flag
        -redraw rect    
    '''
    dragObj.drag = False
    disableResizeButtons(dragObj)
    straightenUpRect(dragObj)
    if dragObj.outRect.w == 0 or dragObj.outRect.h == 0:
        dragObj.active = False
    # endif

    clearCanvasNDraw(dragObj)


# enddef

def disableResizeButtons(dragObj):
    '''
    Description: disable all resize drag handles
    '''
    dragObj.TL = dragObj.TM = dragObj.TR = False
    dragObj.LM = dragObj.RM = False
    dragObj.BL = dragObj.BM = dragObj.BR = False
    dragObj.hold = False


# enddef

def straightenUpRect(dragObj):
    '''
    Description: correct inverted rectangle (user drags R edge over L edge, or T edge over B edge) 
    '''
    if dragObj.outRect.w < 0:
        dragObj.outRect.x = dragObj.outRect.x + dragObj.outRect.w
        dragObj.outRect.w = -dragObj.outRect.w
    # endif
    if dragObj.outRect.h < 0:
        dragObj.outRect.y = dragObj.outRect.y + dragObj.outRect.h
        dragObj.outRect.h = -dragObj.outRect.h
    # endif


# enddef

def clearCanvasNDraw(dragObj):
    '''
    Description:
        -Draw user defined rectangle on image
        -Draw resize handles on rectangle
        -Refresh the window
        -Wait for the next mouse click
    '''
    tmp = dragObj.image_in.copy()
    cv2.rectangle(tmp, (dragObj.outRect.x, dragObj.outRect.y),
                  (dragObj.outRect.x + dragObj.outRect.w,
                   dragObj.outRect.y + dragObj.outRect.h), (0, 255, 0), 1)
    drawSelectMarkers(tmp, dragObj)
    dragObj.image_runtime=tmp
    # cv2.imshow(dragObj.wname, tmp)
    # cv2.waitKey(1)


# enddef

def drawSelectMarkers(image, dragObj):
    # # Top-Left
    # cv2.rectangle(image, (dragObj.outRect.x - dragObj.sBlk, dragObj.outRect.y - dragObj.sBlk),(int(dragObj.outRect.x - dragObj.sBlk + dragObj.sBlk * 2), int(dragObj.outRect.y - dragObj.sBlk + dragObj.sBlk * 2) ),(0, 255, 0), 1)
    # # Top-Right
    # cv2.rectangle(image, (dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk, dragObj.outRect.y - dragObj.sBlk), ( int(dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk + dragObj.sBlk * 2), int(dragObj.outRect.y - dragObj.sBlk + dragObj.sBlk * 2)), (0, 255, 0), 1)
    # # Bottom-Left
    # cv2.rectangle(image, (dragObj.outRect.x - dragObj.sBlk, dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk), (int(dragObj.outRect.x - dragObj.sBlk + dragObj.sBlk * 2), int(dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk + dragObj.sBlk * 2)), (0, 255, 0), 1)
    # # Bottom-Right
    # cv2.rectangle(image, (dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk, dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk), (int(dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk + dragObj.sBlk * 2), int(dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk + dragObj.sBlk * 2)), (0, 255, 0), 1)
    # # Top-Mid
    # cv2.rectangle(image, (int(dragObj.outRect.x + dragObj.outRect.w / 2 - dragObj.sBlk), dragObj.outRect.y - dragObj.sBlk), (int(dragObj.outRect.x + dragObj.outRect.w / 2 - dragObj.sBlk + dragObj.sBlk * 2), int(dragObj.outRect.y - dragObj.sBlk + dragObj.sBlk * 2)), (0, 255, 0), 1)
    # # Bottom-Mid
    # cv2.rectangle(image, (int(dragObj.outRect.x + dragObj.outRect.w / 2 - dragObj.sBlk), dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk), (int(dragObj.outRect.x + dragObj.outRect.w / 2 - dragObj.sBlk + dragObj.sBlk * 2), int(dragObj.outRect.y + dragObj.outRect.h - dragObj.sBlk + dragObj.sBlk * 2)), (0, 255, 0), 1)
    # # Left-Mid
    # cv2.rectangle(image, (dragObj.outRect.x - dragObj.sBlk, int(dragObj.outRect.y + dragObj.outRect.h / 2 - dragObj.sBlk)), (int(dragObj.outRect.x - dragObj.sBlk + dragObj.sBlk * 2),int(dragObj.outRect.y + dragObj.outRect.h / 2 - dragObj.sBlk + dragObj.sBlk * 2)), (0, 255, 0), 1)
    # # Right-Mid
    # cv2.rectangle(image, (dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk,int(dragObj.outRect.y + dragObj.outRect.h / 2 - dragObj.sBlk)), (int(dragObj.outRect.x + dragObj.outRect.w - dragObj.sBlk + dragObj.sBlk * 2),int(dragObj.outRect.y + dragObj.outRect.h / 2 - dragObj.sBlk + dragObj.sBlk * 2)),(0, 255, 0), 1)
    pass
# enddef