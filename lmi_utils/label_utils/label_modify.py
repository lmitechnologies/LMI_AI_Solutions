# %%
import json
import cv2
import os
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import select
import tty
import termios
import label_utils.opencvdragrect.selectinwindow as selectinwindow
from image_utils.img_resize import resize
import time

WINDOW_NAME = "Label Editor"


def rect_gui(wName, crop, bbox, rect_gui_event, cmd_line_event, image_display):
    # initialize roi
    roi = selectinwindow.dragRect
    image = crop
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    (ulx, uly, lrx, lry) = bbox
    w = lrx-ulx
    h = lry-uly
    selectinwindow.init(roi, image, wName, imageWidth,
                        imageHeight, ulx, uly, w, h)
    # set mouse callbacks
    cv2.setMouseCallback(roi.wname, selectinwindow.dragrect, roi)
    roi.returnflag = False
    # update runtime image every 2ms.  Mutable runtime image passed to main thread
    while (not rect_gui_event.is_set()) and (not cmd_line_event.is_set()):
        time.sleep(0.002)
        image_display['runtime_image'] = roi.image_runtime
        if roi.returnflag == True:
            rect_gui_event.set()
    # extract bounding box parameters
    x = roi.outRect.x
    y = roi.outRect.y
    w = roi.outRect.w
    h = roi.outRect.h
    del roi
    # return bounding box to main thread for json update
    bbox = np.array((x, y, x+w, y+h))
    return bbox

# helper function for non-blocking command line input


def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

# non-blocking command line input


def cmd_line(change_str, replacement_labels, current_label, rect_gui_event, cmd_line_event):
    quit = False
    delete_label = False
    change_label = current_label
    old_settings = termios.tcgetattr(sys.stdin)
    print(change_str)
    try:
        tty.setcbreak(sys.stdin.fileno())
        # run while no gui-based or command-based exit flags
        while (not rect_gui_event.is_set()) and (not cmd_line_event.is_set()):
            if isData():
                k = sys.stdin.read(1)
                # quit all editting and generate new json file
                if k == 'q':
                    quit = True
                    cmd_line_event.set()
                    break
                # delete roi - not supported
                elif k == 'd':
                    print('Delete label not supported.')
                    # delete_label=True
                    # cmd_line_event.set()
                    # break
                # skip, go to next label
                elif k == ' ':
                    print('Moving to next sample.')
                    cmd_line_event.set()
                    break
                # relabel roi
                try:
                    change_label = replacement_labels[int(k)]
                    print(
                        f'Reassigning label:{current_label} to {change_label}')
                    cmd_line_event.set()
                except:
                    print('Invalid input.')

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return (quit, delete_label, change_label)


def check_labels(data_path, input_json_path, output_json_path, target_label, replacement_labels, render_delta, window_width):
    # initialize gui and command line threads
    rect_gui_event = threading.Event()
    cmd_line_event = threading.Event()
    # load initial labels
    with open(input_json_path) as f:
        label_dict = json.load(f)
    # form text string for new labels
    change_str = 'Choose ['
    for i, lab in enumerate(replacement_labels):
        change_str = change_str+f'{str(i)}:{lab},'
    change_str = change_str[0:-1]+'] or "d" to delete'
    print('')

    # loop through all labels, enable edit for the target label
    quit = False
    for key in label_dict:
        if quit:
            break
        item = label_dict[key]
        fname = item['filename']
        if not os.path.isfile(os.path.join(data_path, fname)):
            continue
        img = cv2.imread(os.path.join(data_path, fname))
        (H, W) = img.shape[:2]
        regions = item['regions']
        # iterate over a copy of regions
        # runtime image dictionary is passed to gui thread because imshow() only works in main thread
        image_display = {'runtime_image': None}
        cv2.namedWindow(WINDOW_NAME)
        for region in list(regions):
            if quit:
                break
            current_label = region['region_attributes']['Name']
            if current_label == target_label:
                x0 = region['shape_attributes']['x']
                y0 = region['shape_attributes']['y']
                width = region['shape_attributes']['width']
                height = region['shape_attributes']['height']
                ul = (x0, y0)
                lr = ((x0+width, y0+height))
                # crop roi + border region
                render = img.copy()
                dy0=y0 if (y0-render_delta)<0 else render_delta
                dx0=x0 if (x0-render_delta)<0 else render_delta 
                crop = render[np.maximum(y0-dy0, 0):np.minimum(y0+height+dy0, H),
                              np.maximum(x0-dx0, 0):np.minimum(x0+width+dx0, W)]
                # scale cropped region to improve usability
                [Hc, Wc] = crop.shape[:2]
                crop_scale = resize(crop, window_width)
                bbox = np.array([dx0, dy0, dx0+width, dy0+height])
                bbox_scale_in = np.rint((bbox*window_width/Wc)).astype(int)
                # reset exit flags
                rect_gui_event.clear()
                cmd_line_event.clear()
                # launch edit threads
                print(f'Current file: {fname}')
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_gui = executor.submit(
                        rect_gui, WINDOW_NAME, crop_scale, bbox_scale_in, rect_gui_event, cmd_line_event, image_display)
                    future_cmd = executor.submit(
                        cmd_line, change_str, replacement_labels, current_label, rect_gui_event, cmd_line_event)
                    while True:
                        # update UI
                        if image_display['runtime_image'] is not None:
                            cv2.imshow(
                                WINDOW_NAME, image_display['runtime_image'])
                            cv2.waitKey(4)
                        # extract new parameters when done
                        if (future_gui.done() and future_cmd.done()):
                            bbox_scale_out = future_gui.result()
                            (quit, delete_label, change_label) = future_cmd.result()
                            break
                    if not quit:
                        # undo scaling.. correct for padding larger than image boundaries
                        dy0=y0 if (y0-render_delta)<0 else render_delta
                        dx0=x0 if (x0-render_delta)<0 else render_delta 
                        bbox = np.rint((bbox_scale_out/window_width*Wc) - np.asarray((dx0,dy0,dx0,dy0)) + np.asarray((x0, y0, x0, y0))).astype(int)
                        (ulx, uly, lrx, lry) = bbox
                        # update input dictionary
                        region['shape_attributes']['x'] = int(ulx)
                        region['shape_attributes']['y'] = int(uly)
                        region['shape_attributes']['width'] = int(lrx-ulx)
                        region['shape_attributes']['height'] = int(lry-uly)
                        region['region_attributes']['Name'] = change_label

    # write new labels file
    with open(output_json_path, 'w') as json_file:
        print(
            f'Writing new .json file: {output_json_path}')
        json.dump(label_dict, json_file)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--input_data_path', required=True)
    ap.add_argument('--input_json_path', required=True)
    ap.add_argument('--output_json_path', required=True)
    ap.add_argument('--target_label', required=True)
    ap.add_argument('--replacement_labels', required=True)
    ap.add_argument('--window_width', type=int, default=512)
    ap.add_argument('--render_delta', type=int, default=50)
    args = vars(ap.parse_args())

    input_data_path = args['input_data_path']
    input_json_path = args['input_json_path']
    output_json_path = args['output_json_path']
    target_label = args['target_label']
    replacement_labels = args['replacement_labels']
    window_width = args['window_width']
    render_delta = args['render_delta']

    rlabels = replacement_labels.split(",")

    check_labels(input_data_path, input_json_path, output_json_path,
                 target_label, rlabels, render_delta, window_width)
