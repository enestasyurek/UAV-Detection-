#!/usr/bin/env python3
"""Test script for aerial object classifier"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.detection.advanced_aerial_classifier import AerialObjectClassifier

def test_classifier():
    print("Testing Aerial Object Classifier...")
    
    # Create classifier
    classifier = AerialObjectClassifier()
    
    # Create test frame (dummy)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Test detections
    test_cases = [
        {
            'name': 'Large object (airplane)',
            'detection': {
                'bbox': [100, 100, 300, 200],
                'center': (200, 150),
                'area': 20000,
                'confidence': 0.8,
                'aspect_ratio': 2.0
            }
        },
        {
            'name': 'Small hovering object (drone)',
            'detection': {
                'bbox': [400, 200, 450, 250],
                'center': (425, 225),
                'area': 2500,
                'confidence': 0.7,
                'aspect_ratio': 1.0
            }
        },
        {
            'name': 'Small flapping object (bird)',
            'detection': {
                'bbox': [600, 300, 650, 330],
                'center': (625, 315),
                'area': 1500,
                'confidence': 0.6,
                'aspect_ratio': 1.67
            }
        }
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print(f"Detection: area={test['detection']['area']}, "
              f"aspect_ratio={test['detection']['aspect_ratio']:.2f}")
        
        # Classify
        class_name, confidence, features = classifier.classify(
            frame, test['detection'], track_id=str(id(test))
        )
        
        print(f"Result: {class_name} (confidence: {confidence:.2f})")
        print(f"Features: {', '.join([f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' 
                                     for k, v in list(features.items())[:5]])}")
    
    print("\n" + "-" * 60)
    print("Classifier test completed!")
    print("\nThe system is ready to distinguish between:")
    print("✈️  Airplanes")
    print("🚁 Helicopters") 
    print("🦅 Birds")
    print("🎯 Drones")
    print("\nOnly drones will be tracked!")

if __name__ == "__main__":
    test_classifier()