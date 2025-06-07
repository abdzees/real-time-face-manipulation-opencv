# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 22 13:18:55 2021

# @author: droes
# """

# import numpy as np
# import cv2 # conda install opencv
# from matplotlib import pyplot as plt # conda install matplotlib


# # For students
# def initialize_hist_figure():
#     '''
#     Usually called only once to initialize the hist figure.
#     Do not change the essentials of this function to keep the performance advantages.
#     https://www.youtube.com/watch?v=_NNYI8VbFyY
#     '''
#     fig = plt.figure()
#     ax  = fig.add_subplot(111)
#     ax.set_xlim([-0.5, 255.5])
#     # fixed size (you can normalize your values between 0, 3 or other ranges to never exceed this limit)
#     ax.set_ylim([0,3])
#     fig.canvas.draw()
#     background = fig.canvas.copy_from_bbox(ax.bbox)
#     def_x_line = np.arange(0, 256, 1)
#     # def_y_line = np.zeros(shape=(256,))
#     r_plot = ax.plot(def_x_line, def_x_line, 'r', animated=True)[0]
#     g_plot = ax.plot(def_x_line, def_x_line, 'g', animated=True)[0]
#     b_plot = ax.plot(def_x_line, def_x_line, 'b', animated=True)[0]
    
#     return fig, ax, background, r_plot, g_plot, b_plot



# def update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars):
#     '''
#     Uses the initialized figure to update it accordingly to the new values.
#     Do not change the essentials of this function to keep the performance advantages.
#     '''
#     fig.canvas.restore_region(background)        
#     r_plot.set_ydata(r_bars)        
#     g_plot.set_ydata(g_bars)        
#     b_plot.set_ydata(b_bars)

#     ax.draw_artist(r_plot)
#     ax.draw_artist(g_plot)
#     ax.draw_artist(b_plot)
#     fig.canvas.blit(ax.bbox)
    
    

# def plot_overlay_to_image(np_img, plt_figure):
#     '''
#     Use this function to create an image overlay.
#     You must use a matplotlib figure object.
#     Please consider to keep the figure object always outside code loops (performance hint).
#     Use this function for example to plot the histogram on top of your image.
#     White pixels are ignored (transparency effect)-
#     Do not change the essentials of this function to keep the performance advantages.
#     '''
    
#     rgba_buf = plt_figure.canvas.buffer_rgba()
#     (w, h) = plt_figure.canvas.get_width_height()
#     imga = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h,w,4)[:,:,:3]
    
#     # ignore white pixels
#     plt_indices = np.argwhere(imga < 255)

#     # add only non-white values
#     height_indices = plt_indices[:,0]
#     width_indices = plt_indices[:,1]
    
#     np_img[height_indices, width_indices] = imga[height_indices, width_indices]

#     return np_img



# def plot_strings_to_image(np_img, list_of_string, text_color=(255,0,0), right_space=400, top_space=50):
#     '''
#     Plots the string parameters below each other, starting from top right.
#     Use this function for example to plot the default image characteristics.
#     Do not change the essentials of this function to keep the performance advantages.
#     '''
#     y_start = top_space
#     min_size = right_space
#     line_height = 20
#     (h, w, c) = np_img.shape
#     if w < min_size:
#         raise Exception('Image too small in width to print additional text.')
        
#     if h < top_space + line_height:
#         raise Exception('Image too small in height to print additional text.')
    
#     y_pos = y_start
#     x_pos = w - min_size

#     for text in list_of_string:
#         if y_pos >= h:
#             break
#         # SLOW!
#         np_img = cv2.putText(cv2.UMat(np_img), text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
#         y_pos += line_height

#     if type(np_img) is cv2.UMat:
#         np_img = np_img.get()

#     return np_img

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def plot_strings_to_image(img, text_list, position=(10, 30)):
    """Overlay text statistics on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(text_list):
        y = position[1] + i * 30
        cv2.putText(img, text, (position[0], y), font, 0.7, (255, 255, 255), 2)
    return img

def initialize_face_detector():
    """Initialize Haar cascade for face detection"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def apply_filter_overlay(frame, face_cascade, filter_img_path='filters/dog_nose.png'):
    """Apply transparent PNG overlay on detected faces"""
    try:
        filter_img = cv2.imread(filter_img_path, cv2.IMREAD_UNCHANGED)
        if filter_img is None:
            raise FileNotFoundError(f"Filter image not found at {filter_img_path}")
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            resized_filter = cv2.resize(filter_img, (w, h))
            
            # Alpha blending
            alpha = resized_filter[:, :, 3] / 255.0
            for c in range(0, 3):
                frame[y:y+h, x:x+w, c] = (
                    frame[y:y+h, x:x+w, c] * (1 - alpha) + 
                    resized_filter[:, :, c] * alpha
                )
    except Exception as e:
        print(f"Filter error: {str(e)}")
        # Display error on frame
        cv2.putText(frame, "Filter Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def plot_histogram_overlay(sequence, fig):
    """Convert matplotlib figure to OpenCV overlay"""
    # Convert figure to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    hist_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    hist_img = hist_img.reshape(canvas.get_width_height()[::-1] + (3,))
    
    # Resize to fit on top of frame
    h, w = sequence.shape[:2]
    hist_img = cv2.resize(hist_img, (w, h//4))  # 1/4 of frame height
    
    # Convert RGB to BGR for OpenCV
    hist_img = cv2.cvtColor(hist_img, cv2.COLOR_RGB2BGR)
    
    # Overlay at top of frame
    sequence[:h//4, :] = hist_img
    return sequence

def initialize_hist_figure():
    """Create empty histogram figure"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title('RGB Histogram')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, 256])
    ax.grid(True, alpha=0.3)
    
    # Create empty plots for each channel
    r_line, = ax.plot([], [], 'r-', linewidth=1, label='Red')
    g_line, = ax.plot([], [], 'g-', linewidth=1, label='Green')
    b_line, = ax.plot([], [], 'b-', linewidth=1, label='Blue')
    
    ax.legend()
    fig.tight_layout()
    
    # Store background for fast updates
    fig.canvas.draw()  # DRAW FIRST!
    background = fig.canvas.copy_from_bbox(ax.bbox)
    
    return fig, ax, background, r_line, g_line, b_line

def update_histogram(fig, ax, background, r_line, g_line, b_line, r_hist, g_hist, b_hist):
    """Update the histogram data"""
    # Normalize histograms
    r_hist = r_hist / r_hist.max() * 100
    g_hist = g_hist / g_hist.max() * 100
    b_hist = b_hist / b_hist.max() * 100
    
    # Update plot data
    x = np.arange(256)
    r_line.set_data(x, r_hist)
    g_line.set_data(x, g_hist)
    b_line.set_data(x, b_hist)
    
    # Redraw only the changed elements
    fig.canvas.restore_region(background)
    ax.draw_artist(r_line)
    ax.draw_artist(g_line)
    ax.draw_artist(b_line)
    fig.canvas.blit(ax.bbox)