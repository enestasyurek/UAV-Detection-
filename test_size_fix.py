#!/usr/bin/env python3
"""Test script to verify that size mismatch issues in background subtraction are fixed"""

import cv2
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detection.background_subtractor import BackgroundSubtractor
from detection.edgs_yolov8_detector import EDGSYOLOv8Detector

def test_background_subtractor():
    """Test background subtractor with different sized frames"""
    print("Testing BackgroundSubtractor...")
    
    bg_subtractor = BackgroundSubtractor()
    
    # Create test frames with different sizes
    sizes = [(640, 480), (1280, 720), (640, 480), (1920, 1080), (640, 480)]
    
    for i, (w, h) in enumerate(sizes):
        # Create a random frame
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Add some motion (white rectangle that moves)
        cv2.rectangle(frame, (100 + i*10, 100), (200 + i*10, 200), (255, 255, 255), -1)
        
        print(f"  Processing frame {i+1} with size {w}x{h}")
        
        try:
            result = bg_subtractor.process_frame(frame)
            motion_mask = result['motion_mask']
            enhanced_frame = result['enhanced_frame']
            
            # Verify output sizes
            assert motion_mask.shape[:2] == (h, w), f"Motion mask size mismatch: {motion_mask.shape} != {(h, w)}"
            assert enhanced_frame.shape == frame.shape, f"Enhanced frame size mismatch: {enhanced_frame.shape} != {frame.shape}"
            
            print(f"    ✓ Motion mask shape: {motion_mask.shape}")
            print(f"    ✓ Enhanced frame shape: {enhanced_frame.shape}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return False
    
    print("  ✓ All BackgroundSubtractor tests passed!")
    return True

def test_edgs_detector():
    """Test EDGS detector with different sized frames"""
    print("\nTesting EDGSYOLOv8Detector...")
    
    # Initialize detector with CPU for testing
    detector = EDGSYOLOv8Detector(use_gpu=False, enable_edgs=True)
    
    # Test with different frame sizes
    sizes = [(640, 480), (1280, 720), (800, 600)]
    
    for i, (w, h) in enumerate(sizes):
        # Create a test frame
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Add a fake "drone" (dark square)
        cv2.rectangle(frame, (w//2 - 50, h//2 - 50), (w//2 + 50, h//2 + 50), (50, 50, 50), -1)
        
        print(f"  Processing frame {i+1} with size {w}x{h}")
        
        try:
            detections = detector.detect(frame)
            print(f"    ✓ Detected {len(detections)} objects")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return False
    
    print("  ✓ All EDGSYOLOv8Detector tests passed!")
    return True

def main():
    """Run all tests"""
    print("Testing OpenCV size mismatch fixes...\n")
    
    success = True
    
    # Test background subtractor
    if not test_background_subtractor():
        success = False
    
    # Test EDGS detector
    if not test_edgs_detector():
        success = False
    
    if success:
        print("\n✓ All tests passed! Size mismatch issues have been fixed.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())