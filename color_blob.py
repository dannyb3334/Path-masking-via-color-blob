"""
This module provides a set of functions for detecting color blobs in video frames 
to assist with path detection and navigation. 

It focuses on:
- Identifying and highlighting regions of interest based on specific color ranges.
- Detecting contours and convex hulls of color blobs in navigable areas.
- Extracting dominant floor colors to dynamically adapt detection thresholds.
- Processing video streams to visualize and debug path detection.

Functions:
- _helper: Detects and outlines significant color blobs in a cropped region of a frame.
- process_frame: Processes a single frame to identify navigable path regions using color detection.
- process_video: Processes a video stream frame-by-frame to detect paths and highlight key regions.
- get_floor_colour: Computes dominant colors in a specified region using k-means clustering, aiding adaptive detection.
- region_of_interest: Defines and applies a mask to focus on a triangular path region.
- display_lines: Draws detected path lines on a frame for visualization.

Purpose:
The module is designed for real-time or offline video analysis to detect navigable paths based on color features, 
making it suitable for applications in robotics, autonomous navigation, or mobility assistance.

Dependencies:
- OpenCV (cv2)
- NumPy

Instructions for use:
- Run the module to process a video file specified in the `video_path` variable.
- Press 'q' to quit the program.
- Press 'p' to pause or resume the video processing.
"""


import cv2
import numpy as np
from typing import Tuple, List, Optional

def _helper(frame: np.ndarray, colour_max: np.ndarray, colour_min: np.ndarray) -> np.ndarray:
    """
    Process a frame to identify and highlight contours based on color thresholds.

    Args:
        frame (np.ndarray): Input frame in BGR format.
        colour_max (np.ndarray): Maximum HSV color threshold.
        colour_min (np.ndarray): Minimum HSV color threshold.

    Returns:
        np.ndarray: Processed frame with contours highlighted.
    """
    height, width = frame.shape[:2]
    crop = height // 2
    cropped_frame = frame[crop:]

    mask = cv2.erode(cropped_frame, np.ones((8, 8), np.uint8))
    mask = cv2.dilate(mask, np.ones((32, 32), np.uint8))
    mask = cv2.blur(mask, (15, 15), cv2.BORDER_CONSTANT)

    colour_mask = cv2.inRange(mask, colour_min, colour_max)
    cv2.Canny(colour_mask, 100, 200, colour_mask, 3, True)
    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull) > 1000:
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2, offset=(0, crop))

    return frame

def process_video(source: cv2.VideoCapture, colour_min: np.ndarray = np.array([0, 0, 178], np.uint8), 
               colour_max: np.ndarray = np.array([55, 38, 255], np.uint8)) -> None:
    """
    Process a video frame by frame, applying image processing techniques.

    Args:
        source (cv2.VideoCapture): Video capture object.

    Returns:
        None
    """
    frame_count = 0
    paused = False

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Toggle pause/play
        if key == ord('p'):
            paused = not paused

        # Exit loop on 'q'
        if key == ord('q'):
            break

        if not paused:
            ret, frame = source.read()
            if not ret:
                break

            # Downsample frame for faster processing
            frame = frame[::3, ::3]
            frame_count += 1
            print(f"Processing frame {frame_count}")

            # Process the frame
            processed_frame, mask = process_frame(frame, colour_min, colour_max)

            # Concatenate processed frames vertically
            combined_frame = np.vstack((processed_frame, mask))
            cv2.imshow('DEBUG', combined_frame)

    # Clean up resources
    source.release()
    cv2.destroyAllWindows()


def get_floor_colour(img: np.ndarray, k: int = 1) -> List[int]:
    """
    Determine the dominant floor color using k-means clustering.

    Args:
        img (np.ndarray): Input image.
        k (int): Number of clusters for k-means. Defaults to 1.

    Returns:
        List[int]: Dominant color in the image.
    """
    z = img.reshape((-1, 3)).astype(np.float32)

    # Define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to integer values
    centers = np.uint8(centers)

    # Calculate average color if k > 1, else return the single center
    return centers.mean(axis=0).astype(int).tolist() if k > 1 else centers[0].tolist()


def process_frame(frame: np.ndarray, colour_min: np.ndarray, 
               colour_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process an image frame to identify floor colors and contours.

    Args:
        frame (np.ndarray): Input frame in BGR format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed frame and mask.
    """
    def _enclose_polygons(data: np.ndarray) -> np.ndarray:
        def _fill(row: np.ndarray) -> None:
            indices = np.where(row == 255)[0]
            if len(indices) > 1:
                indices = indices[:len(indices) - 1] if len(indices) % 2 != 0 else indices
                for i in range(0, len(indices), 2):
                    row[indices[i] + 1: indices[i + 1]] = 255

        _fill(data[0])
        _fill(data[-1])
        return data

    # Process frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    crop = height // 2
    cropped_frame = frame[crop:]

    # Morphological operations
    mask = cv2.erode(cropped_frame, np.ones((8, 8), np.uint8))
    mask = cv2.dilate(mask, np.ones((32, 32), np.uint8))
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    # Get floor color and create mask
    h, s, v = get_floor_colour(cropped_frame[crop // 3:, width // 3: width - width // 3][::2, ::2])
    colour_mask = cv2.inRange(mask, 
                            np.array([0, max(0, s - 25), v], np.uint8),
                            np.array([min(255, h + 10), min(255, s + 10), 255], np.uint8))
    cv2.Canny(colour_mask, 100, 200, colour_mask, 7, True)

    # Enclose polygons in the mask
    colour_mask = _enclose_polygons(colour_mask)

    # Find contours and draw them
    contours, _ = cv2.findContours(colour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    for cnt in contours:
        hull = cv2.convexHull(cnt, clockwise=True)
        if cv2.contourArea(hull) > height:
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), -1, offset=(0, crop))

    return frame, cv2.cvtColor(colour_mask, cv2.COLOR_GRAY2BGR)


def region_of_interest(frame: np.ndarray) -> np.ndarray:
    """
    Apply a triangular region of interest mask to the frame.

    Args:
        frame (np.ndarray): Input frame.

    Returns:
        np.ndarray: Frame with region of interest applied.
    """
    height, width = frame.shape[:2]
    triangle = np.array([[(width // 3, height), (width - width // 3, height), (width // 2, 0)]], dtype=np.int32)
    mask = np.zeros_like(frame)
    if len(frame.shape) == 3:  # Check if the frame is colored (3 channels)
        mask = mask[:, :, 0]  # Use a single channel for the mask
    cv2.fillPoly(mask, [triangle], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def display_lines(frame: np.ndarray, lines: Optional[np.ndarray]) -> None:
    """
    Display lines on the frame.

    Args:
        frame (np.ndarray): Input frame.
        lines (Optional[np.ndarray]): Detected lines.

    Returns:
        None
    """
    if lines is not None:
        for line in lines:
            if len(line.shape) == 1:  # Handle shape (N, 4)
                x1, y1, x2, y2 = line
            else:  # Handle shape (N, 1, 4)
                x1, y1, x2, y2 = line.reshape(4)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=10)


if __name__ == "__main__":
    video_path = "toronto.mp4"
    process_video(cv2.VideoCapture(video_path))
