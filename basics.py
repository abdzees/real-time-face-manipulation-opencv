
# ### All other basic functions

import cv2
import numpy as np
from collections import Counter

def calculate_mode(channel):
    """Calculate mode without scipy"""
    counts = Counter(channel.flatten())
    return max(counts.items(), key=lambda x: x[1])[0]

def calculate_basic_stats(frame):
    """Calculate mean, std, max, min, mode for each channel"""
    if len(frame.shape) == 3:
        channels = cv2.split(frame)
    else:
        channels = [frame]
    
    results = []
    for ch in channels:
        results.append({
            'mean': np.mean(ch),
            'std': np.std(ch),
            'max': np.max(ch),
            'min': np.min(ch),
            'mode': calculate_mode(ch)  # Using our custom function
        })
    return results

def apply_histogram_equalization(frame):
    """Apply histogram equalization to each channel"""
    if len(frame.shape) == 3:
        channels = cv2.split(frame)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)
    return cv2.equalizeHist(frame)

def edge_detection(frame):
    """Sobel edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

def linear_transform(frame, alpha=1.5, beta=50):
    """Apply linear transformation: output = alpha*input + beta"""
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def calculate_entropy(channel):
    """Calculate entropy of a single channel"""
    hist = cv2.calcHist([channel], [0], None, [256], [0,256])
    hist = hist / hist.sum() + 1e-10  # Avoid log(0)
    return -np.sum(hist * np.log2(hist))

def calculate_rgb_histogram(frame, bins=256):
    """Calculate histogram for all RGB channels"""
    if len(frame.shape) == 2:  # Grayscale
        hist = cv2.calcHist([frame], [0], None, [bins], [0, 256])
        return hist.flatten(), hist.flatten(), hist.flatten()
    
    # For color images - BGR format in OpenCV
    channels = cv2.split(frame)
    hists = []
    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [bins], [0, 256])
        hists.append(hist.flatten())
    
    # OpenCV uses BGR, so return B, G, R to match RGB order expected by plotting

    return hists[2], hists[1], hists[0]  # R, G, B
