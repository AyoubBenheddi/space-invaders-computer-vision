import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Répertoire pour stocker les données
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def capture_gesture(gesture_label, num_samples=100):
    cap = cv2.VideoCapture(0)
    collected = 0
    all_landmarks = []

    print(f"🖐️ Prépare-toi à faire le geste : {gesture_label}. Tu as 3 secondes...")
    cv2.waitKey(3000)

    print(f"Enregistrement de {num_samples} échantillons pour : {gesture_label}")

    while collected < num_samples:
        success, img = cap.read()
        if not success:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraire les coordonnées des landmarks (x, y, z)
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])
                all_landmarks.append(landmark_list)
                collected += 1
                print(f"{collected}/{num_samples} capturés", end='\r')

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Capture Gesture", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Sauvegarde
    df = pd.DataFrame(all_landmarks)
    output_path = os.path.join(DATA_DIR, f"{gesture_label}.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Données enregistrées dans : {output_path}")


if __name__ == "__main__":
    gesture = input("Nom du geste à capturer (ex: LEFT, RIGHT, FIRE, ENTER) : ").upper()
    capture_gesture(gesture_label=gesture, num_samples=100)
