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

# Mapping des prédictions vers les commandes
gesture_to_command = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "FIRE": "FIRE"
}

# Détection d’un geste
def detect_gesture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                return prediction
    return None

# Envoi des commandes via WebSocket
async def send_gesture_commands():
    uri = "ws://localhost:8765"

    print("🎮 Contrôle gestuel connecté à Space Invaders...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connecté !")
        print("Appuie sur 'e' pour démarrer la partie, 'q' pour quitter.")

        # Démarrage du jeu
        loop = asyncio.get_running_loop()
        while True:
            key = await loop.run_in_executor(None, input, "Commande clavier ('e' pour ENTER) : ")
            key = key.lower()

            if key == "e":
                await websocket.send("ENTER")
                print("📨 Commande envoyée : ENTER")
                break
            elif key == "q":
                print("❌ Sortie.")
                return

        # Lancement webcam + détection
        cap = cv2.VideoCapture(0)
        last_command = ""
        cooldown = 0.5  # en secondes
        last_time = time.time()

        print("🖐️ Détection des gestes en cours...")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gesture = detect_gesture(frame)
            current_time = time.time()

            if gesture and gesture in gesture_to_command:
                command = gesture_to_command[gesture]

                # Anti-spam : n'envoie pas trop souvent la même commande
                if command != last_command or (current_time - last_time) > cooldown:
                    await websocket.send(command)
                    print(f"📨 Commande envoyée : {command}")
                    last_command = command
                    last_time = current_time
                    
                    await asyncio.sleep(0.2)  # Attendre que keydown + keyup soient bien gérés


            cv2.imshow("Contrôle Gestuel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Fermeture du module.")
                break

        cap.release()
        cv2.destroyAllWindows()

# Lancement
if __name__ == "__main__":
    asyncio.run(send_gesture_commands())



"""
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

gesture_to_command = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "FIRE": "FIRE"
}

def detect_gesture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                return prediction
    return None

async def gesture_control():
    uri = "ws://localhost:8765"
    print("🎮 Connexion à Space Invaders...")

    async with websockets.connect(uri) as websocket:
        print("✅ Connecté au jeu !")

        # ÉTAPE 1 — Attente de la touche pour envoyer ENTER
        print("🔹 Appuie sur 'e' pour envoyer ENTER et commencer la détection des gestes.")
        while True:
            key = input(">> ").lower()
            if key == "e":
                await websocket.send("ENTER")
                print("📨 Commande ENTER envoyée. Lancement de la détection gestuelle...")
                break
            else:
                print("⏳ En attente de la touche 'e'...")

        # ÉTAPE 2 — Lancement de la détection gestuelle
        cap = cv2.VideoCapture(0)
        last_command = ""
        cooldown = 1  # secondes
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

if __name__ == "__main__":
    asyncio.run(gesture_control())




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
"""