# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 22 11:59:19 2021

# @author: droes
# """
# import cv2  # Add this import at the top
# import keyboard # pip install keyboard

# from capturing import VirtualCamera
# from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
# from basics import histogram_figure_numba

# def custom_processing(img_source_generator):
#     fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
#     for sequence in img_source_generator:
#         # Call your custom processing methods here!

#         rgb_frame = cv2.cvtColor(sequence, cv2.COLOR_BGR2RGB)

#         if keyboard.is_pressed('h'):
#             print('h pressed')
            
#         # Calculate histogram
#         r_bars, g_bars, b_bars = histogram_figure_numba(rgb_frame)         
        
#         # Update histogram
#         update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
#         # Add histogram overlay
#         processed_frame = plot_overlay_to_image(rgb_frame, fig)
        
#         # Display text
#         display_text_arr = ["Test", "abc"]
#         processed_frame = plot_strings_to_image(processed_frame, display_text_arr)

#         # Convert back to BGR for display
#         display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
#         # Display the processed frame
#         cv2.imshow('Camera Feed', display_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # Yield the RGB frame for virtual camera
#         yield processed_frame

#     cv2.destroyAllWindows()

# def main():
#     width = 1280
#     height = 720
#     fps = 30
    
#     vc = VirtualCamera(fps, width, height)
    
#     vc.virtual_cam_interaction(
#         custom_processing(
#             vc.capture_cv_video(0, bgr_to_rgb=False)
#             # or your window screen
#             # vc.capture_screen()
#         )
#     )

# if __name__ == "__main__":
#     main()

import cv2
import keyboard
from capturing import VirtualCamera
from overlays import *
from basics import *

def custom_processing(img_source_generator):
    # Initialize components
    face_cascade = initialize_face_detector()
    current_mode = 'normal'
    show_stats = False
    # Initialize histogram
    hist_fig, hist_ax, hist_bg, r_plot, g_plot, b_plot = initialize_hist_figure()
    show_histogram = False
    
    for sequence in img_source_generator:
        # Calculate histograms
        r_hist, g_hist, b_hist = calculate_rgb_histogram(sequence)
        # Process keyboard inputs
        if keyboard.is_pressed('1'): current_mode = 'edge'
        elif keyboard.is_pressed('2'): current_mode = 'equalize'
        elif keyboard.is_pressed('3'): current_mode = 'dog_filter'
        elif keyboard.is_pressed('s'): show_stats = not show_stats
        elif keyboard.is_pressed('0'): current_mode = 'normal'
        elif keyboard.is_pressed('4'):  # Linear transform
            sequence = linear_transform(sequence, alpha=1.2, beta=30)
        elif keyboard.is_pressed('5'):  # Show entropy
            entropies = [calculate_entropy(ch) for ch in cv2.split(sequence)]
            entropy_text = f"Entropy: R{entropies[0]:.2f} G{entropies[1]:.2f} B{entropies[2]:.2f}"
            sequence = plot_strings_to_image(sequence, [entropy_text])

        elif keyboard.is_pressed('?'):
            help_text = [
                "Controls:",
                "0: Normal  1: Edges  2: Equalize",
                "3: Dog Filter  4: Linear Transform",
                "5: Entropy  h: Histogram  s: Stats",
                "?: Help  q: Quit"
            ]
            sequence = plot_strings_to_image(sequence, help_text, position=(10, 50))
        
        # Toggle with 'h' key
        elif keyboard.is_pressed('h'):
            show_histogram = not show_histogram
            
        if show_histogram:
            update_histogram(hist_fig, hist_ax, hist_bg, 
                           r_plot, g_plot, b_plot, 
                           r_hist, g_hist, b_hist)
            sequence = plot_histogram_overlay(sequence, hist_fig)
        
        # Apply selected mode
        if current_mode == 'edge':
            sequence = cv2.cvtColor(edge_detection(sequence), cv2.COLOR_GRAY2BGR)
        elif current_mode == 'equalize':
            sequence = apply_histogram_equalization(sequence)
        elif current_mode == 'dog_filter':
            sequence = apply_filter_overlay(sequence, face_cascade, 'filters\dog_nose.png')
        
        # Show statistics if enabled
        if show_stats:
            stats = calculate_basic_stats(sequence)
            stats_text = [
                f"Mean: {stats[0]['mean']:.1f} | {stats[1]['mean']:.1f} | {stats[2]['mean']:.1f}",
                f"Std: {stats[0]['std']:.1f} | {stats[1]['std']:.1f} | {stats[2]['std']:.1f}"
            ]
            sequence = plot_strings_to_image(sequence, stats_text)
        
        # Display
        cv2.imshow('Controls: 0-5 | s=stats | q=quit', sequence)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        yield sequence

if __name__ == "__main__":
    vc = VirtualCamera(fps=30, width=1280, height=720)
    vc.virtual_cam_interaction(
        custom_processing(vc.capture_cv_video(0, bgr_to_rgb=False))
    )