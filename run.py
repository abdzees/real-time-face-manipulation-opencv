# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
import cv2
import keyboard
import time
import numpy as np
from capturing import VirtualCamera
from overlays import *
from basics import *

def custom_processing(img_source_generator):
    """
    Main processing function that handles real-time video processing
    """
    print("Initializing processing components...")
    
    # Initialize components
    face_cascade = initialize_face_detector()
    
    # State variables
    current_mode = 'normal'
    show_stats = False
    show_histogram = False
    show_help = False
    
    # Initialize histogram figure
    try:
        hist_fig, hist_ax, hist_bg, r_plot, g_plot, b_plot = initialize_hist_figure()
        histogram_initialized = True
        print("Histogram initialized successfully")
    except Exception as e:
        print(f"Failed to initialize histogram: {e}")
        histogram_initialized = False
    
    # Keyboard state tracking to prevent multiple triggers
    key_states = {}
    def is_key_pressed_once(key):
        current_state = keyboard.is_pressed(key)
        previous_state = key_states.get(key, False)
        key_states[key] = current_state
        return current_state and not previous_state
    
    print("Processing started. Controls:")
    print("0: Normal | 1: Edges | 2: Equalize | 3: Dog Filter")
    print("4: Linear Transform | 5: Entropy | h: Histogram | s: Stats")
    print("?: Help | q: Quit")
    
    frame_count = 0
    
    try:
        for sequence in img_source_generator:
            if sequence is None:
                continue
                
            frame_count += 1
            original_sequence = sequence.copy()
            
            # Handle keyboard inputs (using single-press detection)
            if is_key_pressed_once('0'): 
                current_mode = 'normal'
                print("Mode: Normal")
            elif is_key_pressed_once('1'): 
                current_mode = 'edge'
                print("Mode: Edge Detection")
            elif is_key_pressed_once('2'): 
                current_mode = 'equalize'
                print("Mode: Histogram Equalization")
            elif is_key_pressed_once('3'): 
                current_mode = 'dog_filter'
                print("Mode: Dog Filter")
            elif is_key_pressed_once('4'): 
                current_mode = 'linear_transform'
                print("Mode: Linear Transform")
            elif is_key_pressed_once('5'): 
                current_mode = 'entropy'
                print("Mode: Entropy Display")
            elif is_key_pressed_once('h'): 
                show_histogram = not show_histogram
                print(f"Histogram: {'ON' if show_histogram else 'OFF'}")
            elif is_key_pressed_once('s'): 
                show_stats = not show_stats
                print(f"Stats: {'ON' if show_stats else 'OFF'}")
            elif is_key_pressed_once('/'): 
                show_help = not show_help
                print(f"Help: {'ON' if show_help else 'OFF'}")
            
            # Apply selected processing mode
            processed_sequence = sequence.copy()
            
            if current_mode == 'edge':
                edge_result = edge_detection(sequence)
                processed_sequence = cv2.cvtColor(edge_result, cv2.COLOR_GRAY2BGR)
                
            elif current_mode == 'equalize':
                processed_sequence = apply_histogram_equalization(sequence)
                
            elif current_mode == 'dog_filter':
                processed_sequence = apply_filter_overlay(sequence, face_cascade, 'filters/dog_nose.png')
                
            elif current_mode == 'linear_transform':
                processed_sequence = linear_transform(sequence, alpha=1.2, beta=30)
                
            elif current_mode == 'entropy':
                try:
                    channels = cv2.split(sequence)
                    entropies = [calculate_entropy(ch) for ch in channels]
                    entropy_text = [
                        f"Entropy Values:",
                        f"Blue: {entropies[0]:.2f}",
                        f"Green: {entropies[1]:.2f}",
                        f"Red: {entropies[2]:.2f}"
                    ]
                    processed_sequence = plot_strings_to_image(processed_sequence, entropy_text)
                except Exception as e:
                    print(f"Entropy calculation error: {e}")
            
            # Add histogram overlay if enabled
            if show_histogram and histogram_initialized:
                try:
                    r_hist, g_hist, b_hist = calculate_rgb_histogram(processed_sequence)
                    update_histogram(hist_fig, hist_ax, hist_bg, 
                                   r_plot, g_plot, b_plot, 
                                   r_hist, g_hist, b_hist)
                    processed_sequence = plot_overlay_to_image(processed_sequence, hist_fig)
                except Exception as e:
                    print(f"Histogram overlay error: {e}")
            
            # Show statistics if enabled
            if show_stats:
                try:
                    stats = calculate_basic_stats(processed_sequence)
                    if len(stats) >= 3:  # BGR channels
                        stats_text = [
                            "Statistics (B|G|R):",
                            f"Mean: {stats[0]['mean']:.1f}|{stats[1]['mean']:.1f}|{stats[2]['mean']:.1f}",
                            f"Std: {stats[0]['std']:.1f}|{stats[1]['std']:.1f}|{stats[2]['std']:.1f}",
                            f"Mode: {stats[0]['mode']}|{stats[1]['mode']}|{stats[2]['mode']}"
                        ]
                        processed_sequence = plot_strings_to_image(processed_sequence, stats_text)
                except Exception as e:
                    print(f"Statistics display error: {e}")
            
            # Show help if enabled
            if show_help:
                help_text = [
                    "=== CONTROLS ===",
                    "0: Normal Mode",
                    "1: Edge Detection", 
                    "2: Histogram Equalization",
                    "3: Dog Filter (needs filter image)",
                    "4: Linear Transform",
                    "5: Entropy Display",
                    "h: Toggle Histogram",
                    "s: Toggle Statistics",
                    "/: Toggle This Help",
                    "q: Quit Application"
                ]
                processed_sequence = plot_strings_to_image(processed_sequence, help_text, 
                                                         right_space=500, top_space=30)
            
            # Show current mode indicator
            mode_text = [f"Mode: {current_mode.upper()}"]
            processed_sequence = plot_strings_to_image(processed_sequence, mode_text, 
                                                     text_color=(0, 255, 0), 
                                                     right_space=200, top_space=30)
            
            # Display the processed frame
            window_title = f'Camera Feed - {current_mode.upper()} | Press / for help'
            cv2.imshow(window_title, processed_sequence)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
                print("Quit command received")
                break
                
            # Yield the processed frame for virtual camera
            yield processed_sequence
            
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Processing error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Processing finished")

def main():
    """Main function to set up and run the application"""
    print("=== Real-time Video Processing Application ===")
    
    # Camera settings
    width = 1280
    height = 720
    fps = 30
    camera_id = 0
    
    print(f"Initializing virtual camera: {width}x{height} @ {fps}fps")
    
    try:
        vc = VirtualCamera(fps, width, height)
        
        # Choose input source
        print("Starting camera capture...")
        img_generator = vc.capture_cv_video(camera_id, bgr_to_rgb=False)
        
        # Alternative: Screen capture
        # img_generator = vc.capture_screen()
        
        # Start processing and virtual camera
        processing_generator = custom_processing(img_generator)
        vc.virtual_cam_interaction(processing_generator)
        
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        print("Make sure your camera is connected and not being used by another application")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("Application closed")

if __name__ == "__main__":
    main()