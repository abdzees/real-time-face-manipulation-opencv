import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance

def initialize_hist_figure():
    '''
    Usually called only once to initialize the hist figure.
    Do not change the essentials of this function to keep the performance advantages.
    https://www.youtube.com/watch?v=_NNYI8VbFyY
    '''
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 1])  # Normalized histogram values
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Real-time RGB Histogram')
    ax.grid(True, alpha=0.3)
    
    # Initialize empty data
    x_data = np.arange(0, 256, 1)
    zero_data = np.zeros(256)
    
    # Create line plots for each channel
    r_plot, = ax.plot(x_data, zero_data, 'r-', linewidth=1.5, alpha=0.7, label='Red')
    g_plot, = ax.plot(x_data, zero_data, 'g-', linewidth=1.5, alpha=0.7, label='Green')
    b_plot, = ax.plot(x_data, zero_data, 'b-', linewidth=1.5, alpha=0.7, label='Blue')
    
    ax.legend(loc='upper right')
    fig.tight_layout()
    
    # Draw the figure to create the background
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    
    return fig, ax, background, r_plot, g_plot, b_plot

def update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars):
    '''
    Uses the initialized figure to update it accordingly to the new values.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    # Normalize the histogram data to 0-1 range for better visualization
    r_norm = r_bars / (np.max(r_bars) + 1e-10)
    g_norm = g_bars / (np.max(g_bars) + 1e-10)
    b_norm = b_bars / (np.max(b_bars) + 1e-10)
    
    # Restore the background
    fig.canvas.restore_region(background)
    
    # Update the data for each plot
    r_plot.set_ydata(r_norm)
    g_plot.set_ydata(g_norm)
    b_plot.set_ydata(b_norm)
    
    # Draw only the updated artists
    ax.draw_artist(r_plot)
    ax.draw_artist(g_plot)
    ax.draw_artist(b_plot)
    
    # Blit the updated region
    fig.canvas.blit(ax.bbox)

def plot_overlay_to_image(np_img, plt_figure):
    '''
    Use this function to create an image overlay.
    You must use a matplotlib figure object.
    Please consider to keep the figure object always outside code loops (performance hint).
    Use this function for example to plot the histogram on top of your image.
    White pixels are ignored (transparency effect)-
    Do not change the essentials of this function to keep the performance advantages.
    '''
    # Convert matplotlib figure to numpy array
    canvas = plt_figure.canvas
    canvas.draw()
    
    # Get the RGBA buffer and convert to numpy array
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img_array = buf.reshape(h, w, 3)
    
    # Convert RGB to BGR for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize the histogram image to fit in the top portion of the main image
    img_h, img_w = np_img.shape[:2]
    hist_h = min(img_h // 3, img_array.shape[0])  # Use 1/3 of image height or actual size
    hist_w = min(img_w, img_array.shape[1])
    
    # Resize histogram image
    hist_resized = cv2.resize(img_array, (hist_w, hist_h))
    
    # Create a semi-transparent overlay
    overlay_region = np_img[0:hist_h, 0:hist_w].copy()
    
    # Apply alpha blending (70% histogram, 30% original image)
    alpha = 0.7
    blended = cv2.addWeighted(hist_resized, alpha, overlay_region, 1-alpha, 0)
    
    # Place the blended image back
    np_img[0:hist_h, 0:hist_w] = blended
    
    return np_img

def plot_strings_to_image(np_img, list_of_string, text_color=(255,255,255), right_space=400, top_space=50):
    '''
    Plots the string parameters below each other, starting from top right.
    Use this function for example to plot the default image characteristics.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    if not list_of_string:
        return np_img
        
    y_start = top_space
    min_size = right_space
    line_height = 25
    font_scale = 0.6
    thickness = 2
    
    h, w = np_img.shape[:2]
    
    if w < min_size:
        print(f'Image width ({w}) too small for text overlay (need {min_size})')
        return np_img
        
    if h < top_space + line_height:
        print(f'Image height ({h}) too small for text overlay')
        return np_img
    
    y_pos = y_start
    x_pos = w - min_size + 10  # Small padding from right edge
    
    # Add semi-transparent background for better text readability
    text_bg_color = (0, 0, 0)  # Black background
    
    for i, text in enumerate(list_of_string):
        if y_pos >= h - 10:  # Leave some margin at bottom
            break
            
        # Get text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw semi-transparent background rectangle
        overlay = np_img.copy()
        cv2.rectangle(overlay, (x_pos - 5, y_pos - text_h - 5), 
                     (x_pos + text_w + 5, y_pos + baseline + 5), text_bg_color, -1)
        np_img = cv2.addWeighted(np_img, 0.7, overlay, 0.3, 0)
        
        # Draw the text
        cv2.putText(np_img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        y_pos += line_height

    return np_img

def initialize_face_detector():
    """Initialize Haar cascade for face detection"""
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        return None

def apply_filter_overlay(frame, face_cascade, filter_img_path='filters/dog_nose.png'):
    """Apply transparent PNG overlay on detected faces"""
    if face_cascade is None:
        return frame
        
    try:
        filter_img = cv2.imread(filter_img_path, cv2.IMREAD_UNCHANGED)
        if filter_img is None:
            raise FileNotFoundError(f"Filter image not found at {filter_img_path}")
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Resize filter to face size
            resized_filter = cv2.resize(filter_img, (w, h))
            
            # Check if filter has alpha channel
            if resized_filter.shape[2] == 4:
                # Alpha blending for RGBA images
                alpha = resized_filter[:, :, 3] / 255.0
                for c in range(0, 3):
                    frame[y:y+h, x:x+w, c] = (
                        frame[y:y+h, x:x+w, c] * (1 - alpha) + 
                        resized_filter[:, :, c] * alpha
                    )
            else:
                # Simple overlay for RGB images
                frame[y:y+h, x:x+w] = resized_filter
                
    except Exception as e:
        print(f"Filter error: {str(e)}")
        # Display error on frame
        cv2.putText(frame, "Filter Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame
