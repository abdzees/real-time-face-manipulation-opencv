# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:41 2021

@author: droes
"""

import pyvirtualcam
import numpy as np
import cv2 # conda install opencv
from PIL import ImageGrab # conda install pillow
from matplotlib import pyplot as plt # conda install matplotlib
import keyboard
import time

class VirtualCamera:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width = width
        self.height = height
        
    def capture_screen(self, plt_inside=False, alt_width=0, alt_height=0):
        '''
        Represents the content of the primary monitor.
        Can be used to quickly test your application.
        '''
        
        width = alt_width if alt_width > 0 else self.width
        height = alt_height if alt_height > 0 else self.height
        
        print(f"Starting screen capture at {width}x{height}")
        
        while True:
            try:
                # grab is a slow method!
                img = ImageGrab.grab(bbox=(0, 0, width, height)) #x, y, w, h
                img_np = np.array(img)
                
                # Convert RGBA to RGB if necessary
                if img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                
                # Convert RGB to BGR for OpenCV compatibility
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Resize if dimensions don't match
                if img_np.shape[1] != self.width or img_np.shape[0] != self.height:
                    img_np = cv2.resize(img_np, (self.width, self.height))
                
                if plt_inside:
                    plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
                    
                yield img_np
                
            except Exception as e:
                print(f"Screen capture error: {e}")
                # Yield a black frame as fallback
                yield np.zeros((self.height, self.width, 3), dtype=np.uint8)

            
    def capture_cv_video(self, camera_id, bgr_to_rgb=False):
        '''
        Establishes the connection to the camera via opencv
        Source: https://github.com/letmaik/pyvirtualcam/blob/master/samples/webcam_filter.py
        '''
        print(f"Initializing camera {camera_id}...")
        cv_vid = cv2.VideoCapture(camera_id)

        if not cv_vid.isOpened():
            raise RuntimeError(f'Camera {camera_id} cannot be opened. Check if camera is available.')
            
        # Set camera properties
        cv_vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cv_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv_vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cv_vid.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Additional settings for better performance
        cv_vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay

        # Get actual camera settings
        actual_width = int(cv_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cv_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cv_vid.get(cv2.CAP_PROP_FPS)
        
        print(f'Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps')
        print('Press "q" to quit camera stream')
        
        frame_count = 0
        last_time = time.time()
        
        try:
            while True:
                ret, frame = cv_vid.read()
                if not ret:
                    print("Warning: Failed to read frame from camera")
                    # Yield a black frame as fallback
                    yield np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    continue
                
                # Resize frame to match desired dimensions if necessary
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Convert color space if requested
                if bgr_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check for quit command
                if keyboard.is_pressed('q'):
                    print("Quit command received")
                    break
                
                # FPS monitoring (optional)
                frame_count += 1
                if frame_count % 30 == 0:  # Every 30 frames
                    current_time = time.time()
                    actual_fps_calculated = 30 / (current_time - last_time)
                    print(f"Actual processing FPS: {actual_fps_calculated:.1f}")
                    last_time = current_time
                    
                yield frame
                
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        except Exception as e:
            print(f"Camera capture error: {e}")
        finally:
            print("Releasing camera...")
            cv_vid.release()

    
    def virtual_cam_interaction(self, img_generator, print_fps=True):
        '''
        Provides a virtual camera.
        img_generator must represent a function that acts as a generator and returns image data.
        '''
        print(f'Starting virtual camera: {self.width}x{self.height} @ {self.fps}fps')
        print('Quit camera stream with "q"')
        
        try:
            with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, print_fps=print_fps) as cam:
                for img in img_generator:
                    if img is None:
                        continue
                        
                    # Ensure image is in correct format (RGB for virtual camera)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        # If image is BGR, convert to RGB for virtual camera
                        if hasattr(img, 'dtype') and img.dtype == np.uint8:
                            # Assume BGR input, convert to RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        else:
                            img_rgb = img
                    else:
                        print(f"Warning: Unexpected image format {img.shape}")
                        continue
                    
                    # Ensure correct dimensions
                    if img_rgb.shape[1] != self.width or img_rgb.shape[0] != self.height:
                        img_rgb = cv2.resize(img_rgb, (self.width, self.height))
                    
                    # Provide the image to virtual camera
                    cam.send(img_rgb)
                    # Wait for next frame (fps dependent)
                    cam.sleep_until_next_frame()
                    
        except KeyboardInterrupt:
            print("Virtual camera stopped by user")
        except Exception as e:
            print(f"Virtual camera error: {e}")
        finally:
            print("Virtual camera closed")