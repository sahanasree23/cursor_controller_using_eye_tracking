import cv2
import mediapipe as mp
import pyautogui
import pygame
import string
from scipy.spatial import distance
import sys

# Initialize Pygame
pygame.init()

# Set screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Virtual Keyboard")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Define keyboard layout with space and enter buttons
keyboard_layout = [
    list(string.ascii_uppercase),
    [" ", "@", "_", ".", ",", ":", ";", "'", '"', " ", "\n"],  # Add space and enter buttons
    list("1234567890"),
]

# Define button dimensions
KEY_WIDTH = SCREEN_WIDTH // 15  # Adjusted width for larger buttons
KEY_HEIGHT = SCREEN_HEIGHT // 5

# Define font
font = pygame.font.Font(None, 36)

# Add a function to handle quitting the program
def quit_program():
    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

# Function to draw keyboard buttons
def draw_keyboard():
    for row in range(len(keyboard_layout)):
        for col in range(len(keyboard_layout[row])):
            key = keyboard_layout[row][col]
            x = col * KEY_WIDTH
            y = row * KEY_HEIGHT
            pygame.draw.rect(screen, WHITE, (x, y, KEY_WIDTH, KEY_HEIGHT))
            pygame.draw.rect(screen, BLACK, (x, y, KEY_WIDTH, KEY_HEIGHT), 2)
            text_surface = font.render(key, True, BLACK)
            text_rect = text_surface.get_rect(center=(x + KEY_WIDTH // 2, y + KEY_HEIGHT // 2))
            screen.blit(text_surface, text_rect)



# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Main loop
cam = cv2.VideoCapture(0)  # Initialize the camera
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Flag to track if the virtual keyboard is active
virtual_keyboard_active = False

# Add calibration points and their positions
calibration_points = [(100, 100), (SCREEN_WIDTH - 100, 100), (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)]
calibration_index = 0
calibration_positions = []

# Flag to indicate if calibration is in progress
calibration_in_progress = True

# Number of samples to collect for each calibration point
NUM_SAMPLES_PER_POINT = 50

while True:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_program()
        elif event.type == pygame.KEYDOWN:
            # Check if the 'q' key is pressed
            if event.key == pygame.K_q:
                quit_program()
        elif event.type == pygame.MOUSEBUTTONDOWN and virtual_keyboard_active:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            col = mouse_x // KEY_WIDTH
            row = mouse_y // KEY_HEIGHT
            st=""
            if row < len(keyboard_layout) and col < len(keyboard_layout[row]):
                key_pressed = keyboard_layout[row][col]
                print(key_pressed,end="")
                 # Handle key press

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    # Check if there's a text typing option visible on the screen (for demonstration, we'll use a simple condition)
    text_typing_option_visible = True  # Replace this with your logic to detect text typing option

    if text_typing_option_visible and not virtual_keyboard_active:
        # Open the virtual keyboard if it's not already active
        virtual_keyboard_active = True
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # Ensure the Pygame window is visible

    elif not text_typing_option_visible and virtual_keyboard_active:
        # Close the virtual keyboard if it's active but there's no text typing option
        virtual_keyboard_active = False
        pygame.display.iconify()  # Minimize the Pygame window

    if virtual_keyboard_active:
        # Draw virtual keyboard
        screen.fill(GRAY)
        draw_keyboard()
        pygame.display.flip()

    if calibration_in_progress:
        # Draw calibration points
        for point in calibration_points:
            cv2.circle(frame, point, 10, (0, 0, 255), -1)

        if landmark_points:
            landmarks = landmark_points[0].landmark
            left_eye_landmarks = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in range(159, 145, -1)]
            right_eye_landmarks = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in range(386, 380, -1)]

            if len(left_eye_landmarks) >= 6 and len(right_eye_landmarks) >= 6:
                # Calculate average eye position
                eye_x = (left_eye_landmarks[0][0] + right_eye_landmarks[0][0]) / 2
                eye_y = (left_eye_landmarks[0][1] + right_eye_landmarks[0][1]) / 2

                # Record eye position for calibration
                calibration_positions.append((eye_x, eye_y))

                # Move to next calibration point if enough samples are recorded
                if len(calibration_positions) >= NUM_SAMPLES_PER_POINT:
                    calibration_index += 1
                    calibration_positions = []

                    if calibration_index >= len(calibration_points):
                        calibration_in_progress = False
                        print("Calibration complete", file=sys.stdout)

    if landmark_points and not calibration_in_progress:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)
        left = [landmarks[145], landmarks[159]]
        left_eye_landmarks = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in range(159, 145, -1)]
        right_eye_landmarks = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in range(386, 380, -1)]

        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
        if len(left_eye_landmarks) >= 6 and len(right_eye_landmarks) >= 6:
            # Calculate eye aspect ratio (EAR) for left and right eyes
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)

            # Check if either eye is closed
            if left_ear < 0.25 and right_ear < 0.25:  # Adjusted threshold value
                blink_counter += 1
                if blink_counter >= 1:

                    pyautogui.click()  # Simulate a mouse click
            else:
                blink_counter = 0

    cv2.imshow('Eye Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit_program()

cam.release()
cv2.destroyAllWindows()
