import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import keyboard  # Import the keyboard library

# --- Configuration ---
# Screen dimensions for mapping hand coordinates
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Smoothing factor for cursor movement (0.0 to 1.0, higher is smoother but more lag)
SMOOTHING_FACTOR = 0.2

# Threshold for detecting a "pinch" gesture (distance between finger tips)
# This value might need adjustment based on your camera and hand size.
CLICK_THRESHOLD = 30  # Pixels

# Time window for detecting a double click (in seconds) - Less relevant with dedicated gesture, but kept for clarity
DOUBLE_CLICK_TIME_WINDOW = 0.3

# --- Camera Mapping Configuration ---
# Percentage of the camera frame to use as a buffer on each side.
# Hand movements within the central (1 - 2*BUFFER_PERCENTAGE) region will map to the whole screen.
# Adjust these values (0.0 to 0.5) to find the sweet spot for your camera and comfort.
# E.g., 0.1 means 10% buffer on left, 10% on right, so 80% of camera width maps to screen.
CAMERA_X_BUFFER_PERCENTAGE = 0.15  # 15% buffer on left/right
CAMERA_Y_BUFFER_PERCENTAGE = 0.20  # 20% buffer on top/bottom

# --- Global Variables ---
is_active = False  # Flag to control application's active state
last_click_time = 0  # Used for single click debounce, not double click anymore
last_cursor_x, last_cursor_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
is_dragging = False  # Flag to track drag state
is_clicking = False  # Flag to indicate if ANY click gesture is currently being held (freezes cursor)
is_left_clicking_gesture_held = False  # Tracks if left click gesture is actively held
is_right_clicking_gesture_held = False  # Tracks if right click gesture is actively held
is_double_clicking_gesture_held = (
    False  # Tracks if double click gesture is actively held
)
has_window_been_shown = (
    False  # NEW: Flag to track if the OpenCV window has been successfully displayed
)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Helper Functions ---


def get_distance(point1, point2):
    """Calculates Euclidean distance between two MediaPipe landmarks."""
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5


def get_pixel_coordinates(landmark, image_width, image_height):
    """Converts normalized landmark coordinates to pixel coordinates."""
    return int(landmark.x * image_width), int(landmark.y * image_height)


def update_cursor_position(target_x, target_y):
    """Smoothly updates the mouse cursor position, but only if not clicking."""
    global last_cursor_x, last_cursor_y
    global is_clicking  # Access the global flag

    # Only update cursor position if no click gesture is being held
    if not is_clicking:
        # Apply exponential moving average for smoothing
        smoothed_x = last_cursor_x + (target_x - last_cursor_x) * SMOOTHING_FACTOR
        smoothed_y = last_cursor_y + (target_y - last_cursor_y) * SMOOTHING_FACTOR

        pyautogui.moveTo(smoothed_x, smoothed_y)
        last_cursor_x, last_cursor_y = smoothed_x, smoothed_y


def handle_left_click(hand_landmarks, img_w, img_h):
    """Detects left click gesture (thumb and index finger pinch)."""
    global is_dragging, is_clicking, is_left_clicking_gesture_held

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_x, thumb_y = get_pixel_coordinates(thumb_tip, img_w, img_h)
    index_x, index_y = get_pixel_coordinates(index_finger_tip, img_w, img_h)

    distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

    if distance < CLICK_THRESHOLD:
        # Gesture detected (pinch)
        if (
            not is_left_clicking_gesture_held
        ):  # Only trigger on first detection of gesture
            if not is_dragging:  # If not already dragging, initiate drag or click
                pyautogui.mouseDown()
                is_dragging = True
                print("Left Click Gesture Held (Potential Drag)")
            is_left_clicking_gesture_held = True
            is_clicking = True  # Set overall clicking flag
    else:
        # Gesture released
        if is_left_clicking_gesture_held:  # Only react if gesture was previously held
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                print("Left Click Gesture Released (Item Dropped)")
            else:
                # This path should ideally not be hit if mouseDown is always called on first detection
                # but as a fallback, ensure a click if no drag was active
                pyautogui.click()
                print("Single Click Detected (Fallback)")
            is_left_clicking_gesture_held = False
            is_clicking = False  # Release overall clicking flag


def handle_right_click(hand_landmarks, img_w, img_h):
    """Detects right click gesture (thumb and middle finger pinch)."""
    global is_clicking, is_right_clicking_gesture_held

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    thumb_x, thumb_y = get_pixel_coordinates(thumb_tip, img_w, img_h)
    middle_x, middle_y = get_pixel_coordinates(middle_finger_tip, img_w, img_h)

    distance = ((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2) ** 0.5

    if distance < CLICK_THRESHOLD:
        if not is_right_clicking_gesture_held:  # Only trigger on first detection
            pyautogui.rightClick()
            print("Right Click Detected!")
            is_right_clicking_gesture_held = True
            is_clicking = True  # Set overall clicking flag
            time.sleep(0.5)  # Small delay to prevent multiple right clicks
    else:
        is_right_clicking_gesture_held = False
        is_clicking = False  # Release overall clicking flag


def handle_double_click(hand_landmarks, img_w, img_h):
    """Detects double click gesture (middle finger and ring finger pinch)."""
    global is_clicking, is_double_clicking_gesture_held

    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

    middle_x, middle_y = get_pixel_coordinates(middle_finger_tip, img_w, img_h)
    ring_x, ring_y = get_pixel_coordinates(ring_finger_tip, img_w, img_h)

    distance = ((middle_x - ring_x) ** 2 + (middle_y - ring_y) ** 2) ** 0.5

    if distance < CLICK_THRESHOLD:
        if not is_double_clicking_gesture_held:  # Only trigger on first detection
            pyautogui.doubleClick()
            print("Double Click Detected!")
            is_double_clicking_gesture_held = True
            is_clicking = True  # Set overall clicking flag
            time.sleep(0.5)  # Small delay to prevent multiple double clicks
    else:
        is_double_clicking_gesture_held = False
        is_clicking = False  # Release overall clicking flag


def toggle_active_state():
    """Callback function to toggle the application's active state."""
    global is_active
    is_active = not is_active
    print(f"Application {'Activated' if is_active else 'Deactivated'}!")


def keyboard_listener():
    """Sets up the hotkey listener."""
    # Register the hotkey to call toggle_active_state when pressed
    keyboard.add_hotkey("ctrl+alt+m", toggle_active_state)
    pass  # The thread just needs to set up the hotkey and then can finish its execution.


# --- Main Application Logic ---


def main():
    global is_active, last_cursor_x, last_cursor_y, is_dragging, is_clicking
    global \
        is_left_clicking_gesture_held, \
        is_right_clicking_gesture_held, \
        is_double_clicking_gesture_held
    global has_window_been_shown  # Access the new global flag

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    print("Press Ctrl + Alt + M to activate/deactivate the hand gesture mouse.")
    print("Close the camera window or press Ctrl+C in terminal to exit.")
    print("\nGesture Controls:")
    print("- Cursor Movement: Move index finger.")
    print("- Left Click: Pinch thumb and index finger.")
    print("- Right Click: Pinch thumb and middle finger.")
    print("- Double Click: Pinch middle finger and ring finger.")
    print("- Drag and Drop: Hold left-click gesture, move, then release.")

    cap = None  # Initialize cap outside the loop

    with mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        while True:
            # Always process OpenCV events with waitKey
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break  # Break the loop if 'q' is pressed

            if is_active:
                # Check if the window was closed by the user manually *before* trying to display a new frame
                # This ensures we react immediately to the closure.
                if (
                    has_window_been_shown
                    and cv2.getWindowProperty(
                        "Hand Gesture Mouse (Press Ctrl+Alt+M to toggle)",
                        cv2.WND_PROP_VISIBLE,
                    )
                    < 1
                ):
                    print("Camera window closed by user. Deactivating application.")
                    is_active = False  # Deactivate the application
                    # Continue to the next loop iteration, which will then fall into the 'else' block for cleanup.
                    continue

                if cap is None:
                    # Attempt to open camera only when active
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print(
                            "Error: Could not open webcam. Make sure it's connected and not in use."
                        )
                        is_active = False  # Deactivate if camera fails
                        continue  # Try again in next loop iteration

                    # Robust camera read to ensure stream is ready
                    read_attempts = 0
                    max_read_attempts = (
                        30  # Try to read for up to 30 frames (approx 0.15s)
                    )
                    successful_read = False
                    while read_attempts < max_read_attempts:
                        success, _ = (
                            cap.read()
                        )  # Read a frame to check if camera is live
                        if success:
                            successful_read = True
                            break
                        read_attempts += 1
                        time.sleep(0.005)  # Small delay between attempts

                    if not successful_read:
                        print(
                            f"Error: Camera stream not stable after {max_read_attempts} attempts. Deactivating."
                        )
                        if cap is not None:
                            cap.release()
                            cap = None
                        is_active = False
                        continue  # Skip to next loop iteration

                    # Reset window shown flag when camera is re-opened
                    has_window_been_shown = False

                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame or camera disconnected.")
                    # If camera suddenly stops working, deactivate
                    is_active = False
                    if cap is not None:
                        cap.release()
                        cap = None
                    continue

                # Flip the image horizontally for a natural selfie-view display.
                image = cv2.flip(image, 1)
                image_height, image_width, _ = image.shape

                # Convert the BGR image to RGB.
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process the image and find hands.
                results = hands.process(image_rgb)

                # Draw the hand annotations on the image.
                image.flags.writeable = True

                # --- Draw the active camera region for user guidance ---
                # Calculate pixel coordinates for the active camera region
                buffer_x_pixels = int(image_width * CAMERA_X_BUFFER_PERCENTAGE)
                buffer_y_pixels = int(image_height * CAMERA_Y_BUFFER_PERCENTAGE)

                start_point_rect = (buffer_x_pixels, buffer_y_pixels)
                end_point_rect = (
                    image_width - buffer_x_pixels,
                    image_height - buffer_y_pixels,
                )
                cv2.rectangle(
                    image, start_point_rect, end_point_rect, (0, 255, 0), 2
                )  # Green rectangle

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),  # Corrected function name
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )  # Corrected function name

                        # Get index finger tip coordinates (normalized 0-1)
                        index_finger_tip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ]

                        # Define the active region within the camera frame (normalized 0 to 1)
                        x_min_cam_norm = CAMERA_X_BUFFER_PERCENTAGE
                        x_max_cam_norm = 1.0 - CAMERA_X_BUFFER_PERCENTAGE
                        y_min_cam_norm = CAMERA_Y_BUFFER_PERCENTAGE
                        y_max_cam_norm = 1.0 - CAMERA_Y_BUFFER_PERCENTAGE

                        # Get the hand's current position within the active camera region
                        # Clamp hand_x and hand_y to be within the active camera region
                        # This prevents division by zero if buffer is too large, and ensures mapping works
                        hand_x_norm = max(
                            x_min_cam_norm, min(index_finger_tip.x, x_max_cam_norm)
                        )
                        hand_y_norm = max(
                            y_min_cam_norm, min(index_finger_tip.y, y_max_cam_norm)
                        )

                        # Normalize hand position relative to the active camera region (0 to 1 within the active region)
                        # This scales the hand's movement within the green box to 0-1 range
                        mapped_x_norm = (hand_x_norm - x_min_cam_norm) / (
                            x_max_cam_norm - x_min_cam_norm
                        )
                        mapped_y_norm = (hand_y_norm - y_min_cam_norm) / (
                            y_max_cam_norm - y_min_cam_norm
                        )

                        # Map these normalized values to the full screen dimensions
                        target_x = int(mapped_x_norm * SCREEN_WIDTH)
                        target_y = int(mapped_y_norm * SCREEN_HEIGHT)

                        # Clamp coordinates to screen boundaries (redundant after mapping, but safe)
                        target_x = max(0, min(target_x, SCREEN_WIDTH - 1))
                        target_y = max(0, min(target_y, SCREEN_HEIGHT - 1))

                        # Update cursor position (only if not clicking)
                        update_cursor_position(target_x, target_y)

                        # Check for click gestures
                        # Order matters: check distinct clicks before resetting general clicking flags
                        handle_left_click(hand_landmarks, image_width, image_height)
                        handle_right_click(hand_landmarks, image_width, image_height)
                        handle_double_click(hand_landmarks, image_width, image_height)

                        # If no specific click gesture is actively held, reset is_clicking
                        if not (
                            is_left_clicking_gesture_held
                            or is_right_clicking_gesture_held
                            or is_double_clicking_gesture_held
                        ):
                            is_clicking = False

                else:
                    # No hands detected, ensure drag and clicking states are reset
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                        print("No hand detected, releasing drag.")
                    is_clicking = False  # Ensure clicking flag is false
                    is_left_clicking_gesture_held = False
                    is_right_clicking_gesture_held = False
                    is_double_clicking_gesture_held = False
                    # print("No hand detected, cursor movement enabled.") # Removed to reduce console spam

                # Display the camera feed with annotations
                cv2.imshow("Hand Gesture Mouse (Press Ctrl+Alt+M to toggle)", image)

                # Robust check for window visibility after initial imshow
                # This ensures the window has actually appeared before we start checking for its closure.
                if not has_window_been_shown:
                    # Give OpenCV a moment to render the window
                    cv2.waitKey(1)  # Process events for 1ms
                    # Check if the window is now visible (property >= 1 means visible)
                    if (
                        cv2.getWindowProperty(
                            "Hand Gesture Mouse (Press Ctrl+Alt+M to toggle)",
                            cv2.WND_PROP_VISIBLE,
                        )
                        >= 1
                    ):
                        has_window_been_shown = True
                    else:
                        # If still not visible after initial imshow and waitKey(1), something is wrong
                        print(
                            "Warning: OpenCV window did not become visible during activation. Deactivating."
                        )
                        is_active = False
                        # Continue to the next loop iteration, which will then fall into the 'else' block for cleanup.
                        continue

            else:  # If not active
                if cap is not None:
                    cap.release()  # Release camera if it was open
                    cap = None
                # Always destroy all OpenCV windows when deactivating
                # This handles cases where the window might have been closed by the user
                cv2.destroyAllWindows()
                has_window_been_shown = False  # Reset flag when inactive
                # Small sleep to prevent busy-waiting when inactive
                time.sleep(0.1)

    # Release resources (this block will be reached after loop breaks)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Application exited.Also returned")


if __name__ == "__main__":
    # Disable PyAutoGUI's fail-safe feature (moving mouse to corner to stop)
    # Be careful with this, as it means you can't easily stop the script by
    # moving the mouse to the corner. Use Ctrl+Alt+M or Ctrl+C in terminal.
    pyautogui.FAILSAFE = False
    main()
