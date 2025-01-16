"""
This script allows the user to interactively adjust HSV color thresholds on a video frame using OpenCV trackbars.
The user can pause the video, adjust the HSV values, and see the effect in real-time.

Usage:
    Run the script and use the trackbars to adjust the HSV values.
    Press 'p' to pause/resume the video.
    Press 'q' to quit the application.
"""

import cv2
import numpy as np

def nothing(x: int) -> None:
    """Callback function for trackbars, does nothing."""
    pass

def create_trackbars(window_name: str) -> None:
    """Create trackbars for adjusting HSV values."""
    cv2.createTrackbar('HMin', window_name, 0, 255, nothing)
    cv2.createTrackbar('SMin', window_name, 0, 255, nothing)
    cv2.createTrackbar('VMin', window_name, 0, 255, nothing)
    cv2.createTrackbar('HMax', window_name, 0, 255, nothing)
    cv2.createTrackbar('SMax', window_name, 0, 255, nothing)
    cv2.createTrackbar('VMax', window_name, 0, 255, nothing)
    cv2.setTrackbarPos('HMax', window_name, 255)
    cv2.setTrackbarPos('SMax', window_name, 255)
    cv2.setTrackbarPos('VMax', window_name, 255)

def get_trackbar_values(window_name: str) -> tuple:
    """Get current positions of all trackbars."""
    hMin = cv2.getTrackbarPos('HMin', window_name)
    sMin = cv2.getTrackbarPos('SMin', window_name)
    vMin = cv2.getTrackbarPos('VMin', window_name)
    hMax = cv2.getTrackbarPos('HMax', window_name)
    sMax = cv2.getTrackbarPos('SMax', window_name)
    vMax = cv2.getTrackbarPos('VMax', window_name)
    return hMin, sMin, vMin, hMax, sMax, vMax

def main(video_path, size_reduction=4) -> None:
    """Main function to run the HSV threshold adjustment tool."""
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('frame')
    create_trackbars('frame')

    paused = False
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):
            print('pause')
            paused = not paused
        if k == ord('q'):
            break
        print(paused)
        if not paused:
            ret, frame = cap.read()
            paused = True
            if not ret:
                break
            frame = frame[::size_reduction, ::size_reduction].astype("uint8")

        hMin, sMin, vMin, hMax, sMax, vMax = get_trackbar_values('frame')
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        if (phMin != hMin) or (psMin != sMin) or (pvMin != vMin) or (phMax != hMax) or (psMax != sMax) or (pvMax != vMax):
            print(f"(hMin = {hMin} , sMin = {sMin}, vMin = {vMin}), (hMax = {hMax} , sMax = {sMax}, vMax = {vMax})")
            phMin, psMin, pvMin, phMax, psMax, pvMax = hMin, sMin, vMin, hMax, sMax, vMax

        cv2.imshow('frame', result)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = 'toronto.mp4'
    main(path, size_reduction=2)