import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import joblib
import time

# Chargement du modèle
model = joblib.load("saved_model.pkl")

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Mapping des gestes vers les commandes
gesture_to_command = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "FIRE": "FIRE"
}

# Détection du geste à partir d'une frame
def detect_gesture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:  # 21 points * 3 coords
                prediction = model.predict([landmarks])[0]
                return prediction
    return None

# Boucle principale avec WebSocket
async def gesture_control():
    uri = "ws://localhost:8765"
    print("🎮 Connexion à Space Invaders...")

    async with websockets.connect(uri) as websocket:
        print("✅ Connecté au jeu !")
        print("👋 Fais un geste devant la caméra pour jouer.")

        cap = cv2.VideoCapture(0)
        last_command = ""
        cooldown = 1  # secondes entre deux commandes similaires
        last_time = time.time()

        while True:
            success, frame = cap.read()
            if not success:
                continue

            gesture = detect_gesture(frame)
            if gesture and gesture in gesture_to_command:
                command = gesture_to_command[gesture]
                current_time = time.time()

                if command != last_command or (current_time - last_time) > cooldown:
                    await websocket.send(command)
                    print(f"📨 Commande envoyée : {command}")
                    last_command = command
                    last_time = current_time

            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("❌ Fin de la session.")
                break

        cap.release()
        cv2.destroyAllWindows()

# Lancement du script
if __name__ == "__main__":
    asyncio.run(gesture_control())
