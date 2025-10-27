"""
Enhanced Tongue Detection Meme Display
Detects: Tongue, Fingers, Teeth, and Normal state
A MediaPipe + OpenCV application that displays different hamster memes
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Window settings
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

# Detection thresholds - adjust these values to change sensitivity
TONGUE_OUT_THRESHOLD = 0.03      # Mouth opening for tongue
TEETH_SHOWING_THRESHOLD = 0.025  # Wide smile showing teeth
FINGER_CONFIDENCE = 0.7          # Hand detection confidence

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

# Initialize MediaPipe Hands for finger detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=FINGER_CONFIDENCE,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

def is_tongue_out(face_landmarks):
    """
    Detect if tongue is out by analyzing mouth opening.
    Returns True if mouth is open wide (tongue likely out).
    """
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    mouth_opening = abs(upper_lip.y - lower_lip.y)
    
    # Debug: Uncomment to see values
    # print(f"Mouth opening: {mouth_opening:.4f}")
    
    return mouth_opening > TONGUE_OUT_THRESHOLD

def is_showing_teeth(face_landmarks):
    """
    Detect if teeth are showing (big smile/grin).
    Checks for wide horizontal mouth opening with moderate vertical opening.
    """
    # Mouth corners
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]
    
    # Mouth center points
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    
    # Calculate mouth width and height
    mouth_width = abs(right_corner.x - left_corner.x)
    mouth_height = abs(upper_lip.y - lower_lip.y)
    
    # Wide smile: width is large, height is moderate (not fully open like tongue)
    is_wide_smile = mouth_width > 0.08
    is_moderate_opening = 0.015 < mouth_height < TONGUE_OUT_THRESHOLD
    
    # Debug: Uncomment to see values
    # print(f"Width: {mouth_width:.4f}, Height: {mouth_height:.4f}")
    
    return is_wide_smile and is_moderate_opening

def count_extended_fingers(hand_landmarks):
    """
    Count how many fingers are extended/raised.
    Returns number of fingers up (0-5).
    """
    # Finger tip landmarks: [8, 12, 16, 20] for index, middle, ring, pinky
    # Thumb tip: 4
    # Palm base: 0
    
    fingers_up = 0
    
    # Check thumb (special case - uses x-coordinate)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    if abs(thumb_tip.x - thumb_base.x) > 0.05:
        fingers_up += 1
    
    # Check other four fingers (use y-coordinate)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]  # Middle joints
    
    for tip, pip in zip(finger_tips, finger_pips):
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y
        
        # If tip is above the middle joint, finger is extended
        if tip_y < pip_y - 0.03:
            fingers_up += 1
    
    return fingers_up

def main():
    """
    Main application loop with multi-detection support.
    """
    
    print("=" * 60)
    print("Enhanced Hamster Meme Display")
    print("=" * 60)
    
    # ========================================================================
    # STEP 1: Load all meme images
    # ========================================================================
    
    images = {
        'normal': 'hamsternormal.png',
        'tongue': 'hamster_tongue.png',
        'finger': 'hamsterfinger.png',
        'teeth': 'hamstermeme.png'
    }
    
    loaded_images = {}
    
    for key, filename in images.items():
        if not os.path.exists(filename):
            print(f"\n[ERROR] {filename} not found!")
            print(f"Please add this image to the project directory.")
            return
        
        img = cv2.imread(filename)
        if img is None:
            print(f"\n[ERROR] Could not load {filename}.")
            print("Please check that the file is a valid PNG image.")
            return
        
        # Resize to window size
        loaded_images[key] = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        print(f"[OK] Loaded {filename}")
    
    print("\n[OK] All images loaded successfully!")
    
    # ========================================================================
    # STEP 2: Initialize webcam
    # ========================================================================
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    
    print("[OK] Webcam initialized!")
    
    # ========================================================================
    # STEP 3: Create display windows
    # ========================================================================
    
    cv2.namedWindow('Camera Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Hamster Meme Output', cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow('Camera Input', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow('Hamster Meme Output', WINDOW_WIDTH, WINDOW_HEIGHT)
    
    print("\n" + "=" * 60)
    print("[OK] Application started successfully!")
    print("=" * 60)
    print("\n[GESTURES]")
    print("  😛 Stick tongue out → Tongue hamster")
    print("  🤘 Show fingers → Finger hamster")
    print("  😁 Big smile (show teeth) → Meme hamster")
    print("  😐 Normal face → Normal hamster")
    print("\n[CONTROLS]")
    print("  Press 'q' to quit")
    print("  Press 'd' to toggle debug info\n")
    
    # Default state
    current_meme = loaded_images['normal'].copy()
    show_debug = False
    
    # ========================================================================
    # STEP 4: Main detection loop
    # ========================================================================
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("\n[ERROR] Could not read frame from webcam.")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face and hands
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        
        # ====================================================================
        # Detect gestures and select appropriate meme
        # ====================================================================
        
        detected_gesture = "No face detected"
        fingers_count = 0
        
        # Check for hand gestures first (highest priority)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                fingers_count = count_extended_fingers(hand_landmarks)
                
                # If 1 or more fingers are up, show finger hamster
                if fingers_count >= 1:
                    current_meme = loaded_images['finger'].copy()
                    detected_gesture = f"FINGER GESTURE! ({fingers_count} fingers)"
                    
                    # Draw hand skeleton on camera feed (optional)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    break  # Found a hand, no need to check face
        
        # If no hand gesture, check face expressions
        elif face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                
                # Priority 1: Tongue out (most distinctive)
                if is_tongue_out(face_landmarks):
                    current_meme = loaded_images['tongue'].copy()
                    detected_gesture = "TONGUE OUT!"
                
                # Priority 2: Showing teeth (big smile)
                elif is_showing_teeth(face_landmarks):
                    current_meme = loaded_images['teeth'].copy()
                    detected_gesture = "TEETH SHOWING!"
                
                # Priority 3: Normal face
                else:
                    current_meme = loaded_images['normal'].copy()
                    detected_gesture = "Normal face"
        
        # No face or hands detected
        else:
            current_meme = loaded_images['normal'].copy()
            detected_gesture = "No face detected"
        
        # ====================================================================
        # Display status on camera feed
        # ====================================================================
        
        # Choose color based on detection
        if "FINGER" in detected_gesture:
            color = (255, 0, 255)  # Magenta for fingers
        elif "TONGUE" in detected_gesture:
            color = (0, 255, 0)    # Green for tongue
        elif "TEETH" in detected_gesture:
            color = (0, 255, 255)  # Yellow for teeth
        elif "Normal" in detected_gesture:
            color = (255, 255, 0)  # Cyan for normal
        else:
            color = (0, 0, 255)    # Red for no detection
        
        cv2.putText(frame, detected_gesture, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Show debug info if enabled
        if show_debug:
            cv2.putText(frame, f"Press 'd' to hide debug", (10, WINDOW_HEIGHT - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if fingers_count > 0:
                cv2.putText(frame, f"Fingers: {fingers_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # ====================================================================
        # Display windows
        # ====================================================================
        
        cv2.imshow('Camera Input', frame)
        cv2.imshow('Hamster Meme Output', current_meme)
        
        # ====================================================================
        # Handle keyboard input
        # ====================================================================
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n[QUIT] Quitting application...")
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[DEBUG] Debug mode: {'ON' if show_debug else 'OFF'}")
    
    # ========================================================================
    # STEP 5: Cleanup
    # ========================================================================
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()
    
    print("[OK] Application closed successfully.")
    print("Thanks for using Enhanced Hamster Meme Display!\n")

if __name__ == "__main__":
    main()