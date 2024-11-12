import time
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def are_both_arms_raised(landmarks, angle_margin=40):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_arm_raised = left_wrist.visibility > 0.5 and left_wrist.y < left_shoulder.y
    right_arm_raised = right_wrist.visibility > 0.5 and right_wrist.y < right_shoulder.y

    if not (left_arm_raised and right_arm_raised):
        return False

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    left_arm_extended = 180 - angle_margin <= left_arm_angle <= 180 + angle_margin
    right_arm_extended = 180 - angle_margin <= right_arm_angle <= 180 + angle_margin

    return left_arm_extended and right_arm_extended

async def person_detection_server(websocket, path):
    global cap, pose, global_arms_raised, arm_raise_start_time, prev_message  # dichiara prev_message come globale
    
    while True:
        ret, frame = cap.read()
        if not ret:
            await sendStatusChanged(websocket, "No camera feed.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            if are_both_arms_raised(results.pose_landmarks.landmark):
                if not global_arms_raised:
                    global_arms_raised = True
                    arm_raise_start_time = time.time()
                elif time.time() - arm_raise_start_time >= 3:
                    await sendStatusChanged(websocket, "Person with arms raised")
                    cv2.imshow("Segmentazione in tempo reale", frame)
                    continue
            else:
                global_arms_raised = False
                arm_raise_start_time = None
                await sendStatusChanged(websocket, "Person detected but arms not raised")
        else:
            await sendStatusChanged(websocket, "No person detected")

        cv2.imshow("Interfaccia", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    

async def sendStatusChanged(websocket, message):
    global prev_message  # dichiara prev_message come globale
    if message != prev_message:
        await websocket.send(message)
        prev_message = message  # aggiorna prev_message se il messaggio è stato inviato
    

async def main():
    global cap, pose, global_arms_raised, arm_raise_start_time, prev_message
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    global_arms_raised = False
    arm_raise_start_time = None
    prev_message = ""  # inizializza prev_message
    
    async with websockets.serve(person_detection_server, "localhost", 6789):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
