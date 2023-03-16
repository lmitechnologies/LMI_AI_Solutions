def bbox_resize(box,  W, H, box_definition='TF_OD'):
        """
        DESCRIPTION:
            bbox_resize() generate absolute coordinates for normalized bounding box
    
        ARGUMENTS:
            box: normalized bounding box coordinates (0->1) (Y_ul, X_ul, Y_lr, X_lr)
            W: image width
            H: image height
            box_definition: TF_OD or YOLO
        
        RETURNS:
            tuple including: (startYnew, startXnew, endYnew, endXnew) where each new coordinate is scaled pixel coordinates
        """
        if box_definition=='TF_OD':
            (startY,startX,endY,endX)=box
        elif box_definition=='YOLO':
            (startX,startY,endX,endY)=box 
        startXnew = int(startX*W)
        startYnew = int(startY*H)
        endXnew = int(endX*W)
        endYnew = int(endY*H)
        
        return (startYnew, startXnew, endYnew, endXnew)
