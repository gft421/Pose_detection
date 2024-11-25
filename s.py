from flask import Flask, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose and Hand Tracking
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

POSITION_TOLERANCE = 0.15
reference_video_path = None


def get_normalized_hand_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = []
        for landmarks in results.multi_hand_landmarks:
            hand = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            center = np.mean(hand[:, :2], axis=0)
            hand[:, :2] -= center
            max_distance = np.max(np.linalg.norm(hand[:, :2], axis=1))
            hand /= max_distance
            hand_landmarks.append(hand)
        return hand_landmarks
    return None


def check_perfect_pose_match(user_hand_landmarks, predefined_hand_landmarks):
    if len(user_hand_landmarks) != len(predefined_hand_landmarks):
        return False
    for i in range(len(user_hand_landmarks)):
        if np.any(np.abs(user_hand_landmarks[i] - predefined_hand_landmarks[i]) > POSITION_TOLERANCE):
            return False
    return True


def detect_finger_gesture(hand_landmarks):
    thumb_up = hand_landmarks[4, 1] < hand_landmarks[3, 1] and hand_landmarks[3, 1] < hand_landmarks[2, 1]
    peace_sign = hand_landmarks[8, 1] < hand_landmarks[7, 1] and hand_landmarks[12, 1] < hand_landmarks[11, 1]
    if thumb_up:
        return "Thumb Up"
    elif peace_sign:
        return "Peace Sign"
    return "Unknown Gesture"


@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    global reference_video_path
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    reference_video_path = file_path
    return jsonify({"message": "Reference video uploaded successfully!", "path": reference_video_path}), 200


@app.route('/compare_poses', methods=['GET'])
def compare_poses():
    if not reference_video_path:
        return jsonify({"error": "No reference video uploaded"}), 400

    cap_video = cv2.VideoCapture(reference_video_path)
    cap_webcam = cv2.VideoCapture(0)

    if not cap_video.isOpened() or not cap_webcam.isOpened():
        return jsonify({"error": "Could not open video or webcam."}), 500

    def generate_frames():
        try:
            while cap_video.isOpened() and cap_webcam.isOpened():
                ret_video, frame_video = cap_video.read()
                ret_webcam, frame_webcam = cap_webcam.read()

                if not ret_video or not ret_webcam:
                    break

                video_hand_landmarks = get_normalized_hand_landmarks(frame_video)
                if video_hand_landmarks:
                    video_hand_landmarks = video_hand_landmarks[0]

                webcam_hand_landmarks = get_normalized_hand_landmarks(frame_webcam)
                if webcam_hand_landmarks:
                    webcam_hand_landmarks = webcam_hand_landmarks[0]

                feedback_message = "Waiting for pose..."
                feedback_color = (255, 255, 255)

                if video_hand_landmarks is None or webcam_hand_landmarks is None:
                    feedback_message = "No valid hand pose detected."
                else:
                    is_matched = check_perfect_pose_match(webcam_hand_landmarks, video_hand_landmarks)
                    feedback_message = "Pose Matched!" if is_matched else "Pose Mismatched!"
                    gesture = detect_finger_gesture(webcam_hand_landmarks)

                cv2.putText(frame_webcam, feedback_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)
                _, buffer = cv2.imencode('.jpg', frame_webcam)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            cap_video.release()
            cap_webcam.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
