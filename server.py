import cv2
import time
import serial
import threading
from flask import Flask, render_template, request, jsonify, Response
from collections import Counter

app = Flask(__name__)

# Connexion série avec l'Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# Initialisation de la vidéo
cap = None
video_recording = False
out = None
nom_fichier = "video_default.avi"

# Liste des stimulations
stimulations = []
stop_stimulations = False

# Fonction pour envoyer les stimulations à l'Arduino
def envoyer_stimulation(couleur, brightness, duration, mode):
    command = f"{couleur},{brightness},{duration},{mode}\n"
    print(f"Envoi de la commande: {command}")
    arduino.write(command.encode())

    while True:
        if arduino.in_waiting > 0:
            response = arduino.readline().decode('utf-8').strip()
            print(f"Réponse Arduino: {response}")
            if response == "done":
                break

# Fonction pour le flux vidéo avec enregistrement
def generate_frames():
    global cap, video_recording, out

    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        combined_frame = cv2.hconcat([frame, gray_frame])

        if video_recording and out is not None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(gray_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out.write(gray_frame)

        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Fonction pour grouper les stimulations similaires
def grouper_stimulations():
    global stimulations
    grouped = Counter(tuple(stim.items()) for stim in stimulations)

    stim_list = []
    for key, count in grouped.items():
        stim_dict = dict(key)
        stim_dict['count'] = count
        stim_list.append(stim_dict)
    return stim_list

@app.route('/')
def index():
    return render_template('index.html', stimulations=stimulations)

@app.route('/ajouter_stimulation', methods=['POST'])
def ajouter_stimulation():
    global stimulations
    couleur = request.form['couleur']
    brightness = int(request.form['brightness'])
    duration = int(request.form['duree'])
    mode = request.form['mode']
    pause = int(request.form['pause'])
    quantite = int(request.form.get('quantite', 1))

    for stim in stimulations:
        if (stim['couleur'] == couleur and stim['brightness'] == brightness and
            stim['duration'] == duration and stim['mode'] == mode and stim['pause'] == pause):
            stim['count'] += quantite
            break
    else:
        stimulations.append({
            'couleur': couleur,
            'brightness': brightness,
            'duration': duration,
            'mode': mode,
            'pause': pause,
            'count': quantite
        })

    return jsonify(stimulations=grouper_stimulations())

@app.route('/supprimer_stimulation', methods=['POST'])
def supprimer_stimulation():
    index = int(request.form['index'])

    if 0 <= index < len(stimulations):
        if stimulations[index]['count'] > 1:
            stimulations[index]['count'] -= 1
        else:
            stimulations.pop(index)

    return jsonify(stimulations=stimulations)

@app.route('/supprimer_ligne_stimulation', methods=['POST'])
def supprimer_ligne_stimulation():
    index = int(request.form['index'])

    if 0 <= index < len(stimulations):
        stimulations.pop(index)  # Supprime toute la ligne

    return jsonify(stimulations=grouper_stimulations())

@app.route('/lancer_stimulations', methods=['POST'])
def lancer_stimulations():
    global stop_stimulations
    stop_stimulations = False

    for stim in stimulations:
        if stop_stimulations:
            break
        for _ in range(stim['count']):
            if stop_stimulations:
                break
            envoyer_stimulation(stim['couleur'], stim['brightness'], stim['duration'], stim['mode'])
            time.sleep(stim['pause'] / 1000)

    return jsonify({"status": "Stimulations lancées!"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video', methods=['POST'])
def video():
    global video_recording, out, nom_fichier

    action = request.form['action']
    nom_video = request.form.get('nom_video', 'video_default').strip()
    nom_fichier = f"{nom_video}.avi"

    if action == "start" and not video_recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(nom_fichier, fourcc, 30.0, (640, 480))
        video_recording = True
        return jsonify({"message": f"Enregistrement démarré : {nom_fichier}"}), 200

    elif action == "stop" and video_recording:
        video_recording = False
        out.release()
        out = None
        return jsonify({"message": f"Enregistrement arrêté : {nom_fichier}"}), 200

    return jsonify({"error": "Action invalide"}), 400

@app.route('/stop_stimulations', methods=['POST'])
def stop_stimulations():
    global stop_stimulations
    stop_stimulations = True
    return jsonify({"status": "Stimulations arrêtées"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
