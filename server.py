import cv2
import time
import serial
import threading
import csv
import os
import io
import signal
from flask import Flask, render_template, request, jsonify, Response
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Generator, Tuple

@dataclass(frozen=True)
class _StimulationConfig:
    color: str
    brightness: int
    duration: int
    mode: str
    trigger_time: int
    video_before: int
    video_after: int

@dataclass(frozen=True)
class _ExperienceConfig:
    name: str
    delay_ms: int
    loop_count: int = 1

class _HardwareState:
    def __init__(self) -> None:
        self.arduino: Optional[serial.Serial] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_recording: bool = False
        self.out: Optional[cv2.VideoWriter] = None
        self.recap_file: Optional[io.TextIOWrapper] = None
        self.recap_writer: Any = None
        self.video_ref_count: int = 0
        self.video_lock: threading.Lock = threading.Lock()
        self.latest_frame: Optional[bytes] = None
        self.camera_lock: threading.Lock = threading.Lock()
        self.global_calculated_fps: float = 15.0
        self.stimulations: List[Dict[str, Any]] = []
        self.stop_stimulations: bool = False
        self.is_running: bool = False

    def _connect_arduino(self) -> None:
        try:
            self.arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        except serial.SerialException:
            self.arduino = None

class _StimulationService:
    def __init__(self, state: _HardwareState) -> None:
        self.__state = state

    def _camera_thread_function(self) -> None:
        self.__state.cap = None
        _frame_times: List[float] = []
        
        while True:
            if self.__state.cap is None or not self.__state.cap.isOpened():
                self.__state.cap = cv2.VideoCapture(0)
                if self.__state.cap.isOpened():
                    self.__state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.__state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.__state.cap.set(cv2.CAP_PROP_FPS, 30)
                    _frame_times = []
                else:
                    time.sleep(1)
                    continue
                    
            _success, _frame = self.__state.cap.read()
            if not _success:
                self.__state.cap.release()
                self.__state.cap = None
                time.sleep(1)
                continue
                
            _now: float = time.time()
            _frame_times.append(_now)
            if len(_frame_times) > 30:
                _frame_times.pop(0)
                
            if len(_frame_times) > 1:
                _elapsed: float = _frame_times[-1] - _frame_times[0]
                if _elapsed > 0:
                    self.__state.global_calculated_fps = (len(_frame_times) - 1) / _elapsed
                
            _gray_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
            _gray_frame = cv2.cvtColor(_gray_frame, cv2.COLOR_GRAY2BGR)
            
            with self.__state.video_lock:
                if self.__state.video_recording and self.__state.out is not None:
                    try:
                        self.__state.out.write(_gray_frame)
                    except cv2.error:
                        pass

            _ret, _buffer = (False, None)
            if not self.__state.is_running:
                _ret, _buffer = cv2.imencode('.jpg', _gray_frame)
            if _ret:
                with self.__state.camera_lock:
                    self.__state.latest_frame = _buffer.tobytes()

    def _send_stimulation(self, color: str, brightness: int, duration: int, mode: str, timing_s: float, video_before: int, video_after: int) -> None:
        _command: str = f"{color},{brightness},{duration},{mode}\n"
        
        if self.__state.recap_writer:
            self.__state.recap_writer.writerow([round(timing_s, 2), color, brightness, duration, mode, video_before, video_after])
            if self.__state.recap_file:
                self.__state.recap_file.flush()

        if self.__state.arduino:
            self.__state.arduino.write(_command.encode())

    def _generate_frames(self) -> Generator[bytes, None, None]:
        while True:
            with self.__state.camera_lock:
                _frame = self.__state.latest_frame
                
            if _frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + _frame + b'\r\n')
            
            time.sleep(0.03)

    def _start_auto_video(self, video_name: str, record_folder: str) -> None:
        with self.__state.video_lock:
            self.__state.video_ref_count += 1
            if self.__state.video_recording:
                return
                
            os.makedirs(record_folder, exist_ok=True)
            _safe_video: str = video_name.replace('/', '_').replace('\\', '_')
            _file_name: str = os.path.join(record_folder, f"{_safe_video}.avi")
            _fps_to_use: float = self.__state.global_calculated_fps if 1.0 < self.__state.global_calculated_fps < 60.0 else 15.0
            
            _fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.__state.out = cv2.VideoWriter(_file_name, _fourcc, round(_fps_to_use, 2), (640, 480))
            self.__state.video_recording = True

    def _stop_auto_video(self) -> None:
        with self.__state.video_lock:
            if self.__state.video_ref_count > 0:
                self.__state.video_ref_count -= 1
            if self.__state.video_ref_count == 0 and self.__state.video_recording:
                self.__state.video_recording = False
                if self.__state.out:
                    self.__state.out.release()
                    self.__state.out = None

    def _run_stimulations(self, experiment_name: str) -> None:
        _start_time: float = time.time()
        
        _timestamp_exp: str = time.strftime("%Y-%m-%d_%H-%M-%S")
        _record_folder: str = f"records/{experiment_name}_{_timestamp_exp}"
        os.makedirs(_record_folder, exist_ok=True)
        
        _csv_name: str = os.path.join(_record_folder, f"{experiment_name}.csv")
        try:
            self.__state.recap_file = open(_csv_name, mode='w', newline='')
            self.__state.recap_writer = csv.writer(self.__state.recap_file)
            self.__state.recap_writer.writerow(["Timing_s", "Color", "Brightness", "Duration_ms", "Mode", "VideoBefore_s", "VideoAfter_s"])
        except OSError:
            self.__state.recap_file = None
            self.__state.recap_writer = None
        
        _actions: List[Dict[str, Any]] = []
        _color_map: Dict[str, str] = {"255,0,0": "Red", "0,255,0": "Green", "0,0,255": "Blue", "255,255,0": "Yellow", "0,255,255": "Cyan", "255,0,255": "Magenta", "255,255,255": "White"}
        
        _grouped_sessions: List[Dict[str, Any]] = []
        _current_session: Optional[Dict[str, Any]] = None
        
        for _stim in self.__state.stimulations:
            _t_target: int = _stim['trigger_time']
            _t_start_video: float = max(0, _t_target - _stim['video_before'])
            _t_end_video: float = _t_target + (_stim['duration'] / 1000.0) + _stim['video_after']
            
            if _current_session is None:
                _current_session = {'start': _t_start_video, 'end': _t_end_video, 'stims': [_stim]}
            else:
                if _t_start_video <= _current_session['end']:
                    _current_session['end'] = max(_current_session['end'], _t_end_video)
                    _current_session['stims'].append(_stim)
                else:
                    _grouped_sessions.append(_current_session)
                    _current_session = {'start': _t_start_video, 'end': _t_end_video, 'stims': [_stim]}
        if _current_session:
            _grouped_sessions.append(_current_session)
            
        for _session in _grouped_sessions:
            _names: List[str] = []
            for _s in _session['stims']:
                _c_name: str = _color_map.get(_s['color'], "Custom")
                _names.append(f"T{_s['trigger_time']}_D{_s['duration']}_{_c_name}")
            _video_name: str = "+".join(_names)
            
            _actions.append({'type': 'start_video', 'time': _start_time + _session['start'], 'name': _video_name, 'folder': _record_folder})
            _actions.append({'type': 'stop_video', 'time': _start_time + _session['end']})
            
        for _stim in self.__state.stimulations:
            _actions.append({'type': 'flash', 'time': _start_time + _stim['trigger_time'], 'stim': _stim})
            
        _actions.sort(key=lambda x: x['time'])

        try:
            for _action in _actions:
                if self.__state.stop_stimulations:
                    break

                while time.time() < _action['time']:
                    if self.__state.stop_stimulations:
                        break
                    time.sleep(0.1)

                if self.__state.stop_stimulations:
                    break

                if _action['type'] == 'start_video':
                    self._start_auto_video(_action['name'], _action['folder'])
                elif _action['type'] == 'flash':
                    _stim = _action['stim']
                    _timing_s: float = time.time() - _start_time
                    self._send_stimulation(_stim['color'], _stim['brightness'], _stim['duration'], _stim['mode'], _timing_s, _stim['video_before'], _stim['video_after'])
                elif _action['type'] == 'stop_video':
                    self._stop_auto_video()
        except Exception:
            import traceback
            print(f"[_run_stimulations] error during '{experiment_name}':")
            traceback.print_exc()
        finally:
            with self.__state.video_lock:
                self.__state.video_ref_count = 0
                if self.__state.video_recording:
                    self.__state.video_recording = False
                    if self.__state.out:
                        self.__state.out.release()
                        self.__state.out = None

            if self.__state.recap_file:
                self.__state.recap_file.close()
                self.__state.recap_file = None
                self.__state.recap_writer = None

    def _execute_with_delay(self, config: _ExperienceConfig) -> None:
        self.__state.stop_stimulations = False
        self.__state.is_running = True

        try:
            if config.delay_ms > 0:
                time.sleep(config.delay_ms / 1000.0)

            if self.__state.stop_stimulations:
                return

            _loop_count: int = max(1, config.loop_count)

            for _i in range(_loop_count):
                if self.__state.stop_stimulations:
                    break
                _run_name: str = config.name if _loop_count == 1 else f"{config.name}_run{_i + 1}"
                try:
                    self._run_stimulations(_run_name)
                except Exception:
                    import traceback
                    print(f"[_execute_with_delay] run '{_run_name}' failed, continuing to next loop iteration:")
                    traceback.print_exc()
        finally:
            self.__state.is_running = False

    def _schedule_experience(self, config: _ExperienceConfig) -> None:
        _thread = threading.Thread(target=self._execute_with_delay, args=(config,))
        _thread.start()
        
    def _start_camera_thread(self) -> None:
        _camera_thread = threading.Thread(target=self._camera_thread_function, daemon=True)
        _camera_thread.start()

def _create_app(service: _StimulationService, state: _HardwareState) -> Flask:
    _app = Flask(__name__)
    
    @_app.route('/')
    def _index() -> str:
        return render_template('index.html', stimulations=state.stimulations)

    @_app.route('/add_stimulation', methods=['POST'])
    def _add_stimulation() -> Response:
        _color: str = request.form['color']
        _brightness: int = int(request.form['brightness'])
        _duration: int = int(request.form['duration'])
        _mode: str = request.form['mode']
        _trigger_time: int = int(request.form['trigger_time'])
        _video_before: int = int(request.form['video_before'])
        _video_after: int = int(request.form['video_after'])

        state.stimulations.append({
            'color': _color,
            'brightness': _brightness,
            'duration': _duration,
            'mode': _mode,
            'trigger_time': _trigger_time,
            'video_before': _video_before,
            'video_after': _video_after
        })
        
        state.stimulations = sorted(state.stimulations, key=lambda x: x['trigger_time'])
        return jsonify(stimulations=state.stimulations)

    @_app.route('/delete_stimulation', methods=['POST'])
    def _delete_stimulation() -> Response:
        _index: int = int(request.form['index'])

        if 0 <= _index < len(state.stimulations):
            state.stimulations.pop(_index)

        return jsonify(stimulations=state.stimulations)

    @_app.route('/edit_stimulation', methods=['POST'])
    def _edit_stimulation() -> Response:
        _index: int = int(request.form['index'])
        _field: str = request.form['field']
        _value: str = request.form['value']
        _val: Any = _value

        if 0 <= _index < len(state.stimulations):
            if _field in ['brightness', 'duration', 'trigger_time', 'video_before', 'video_after']:
                _val = int(_value)
            state.stimulations[_index][_field] = _val
        return jsonify(stimulations=state.stimulations)

    @_app.route('/move_stimulation', methods=['POST'])
    def _move_stimulation() -> Response:
        _index: int = int(request.form['index'])
        _direction: str = request.form['direction']

        if _direction == 'up' and _index > 0:
            state.stimulations[_index], state.stimulations[_index-1] = state.stimulations[_index-1], state.stimulations[_index]
        elif _direction == 'down' and _index < len(state.stimulations) - 1:
            state.stimulations[_index], state.stimulations[_index+1] = state.stimulations[_index+1], state.stimulations[_index]

        return jsonify(stimulations=state.stimulations)

    @_app.route('/import_csv', methods=['POST'])
    def _import_csv() -> Tuple[Response, int]:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
            
        _file = request.files['file']
        if _file.filename == '':
            return jsonify({"error": "Empty file"}), 400
            
        if _file:
            _content: str = _file.stream.read().decode("UTF8")
            _first_line: str = _content.split('\n', 1)[0]
            _delimiter: str = ';' if _first_line.count(';') > _first_line.count(',') else ','
            
            _stream = io.StringIO(_content, newline=None)
            _csv_input = csv.reader(_stream, delimiter=_delimiter)
            next(_csv_input, None)
            
            state.stimulations = []
            for _row in _csv_input:
                if not _row or len(_row) < 5: continue
                try:
                    _timing_s: float = float(_row[0])
                    _temps_int: int = int(round(_timing_s))
                    _color: str = _row[1]
                    _brightness: int = int(_row[2])
                    _duration: int = int(_row[3])
                    _mode: str = _row[4]
                    
                    _v_before: int = int(_row[5]) if len(_row) > 5 else 20
                    _v_after: int = int(_row[6]) if len(_row) > 6 else 20
                    
                    state.stimulations.append({
                        'color': _color,
                        'brightness': _brightness,
                        'duration': _duration,
                        'mode': _mode,
                        'trigger_time': _temps_int,
                        'video_before': _v_before,
                        'video_after': _v_after
                    })
                except ValueError:
                    pass
                    
            return jsonify(stimulations=state.stimulations), 200
        return jsonify({"error": "Error"}), 400

    @_app.route('/start_stimulations', methods=['POST'])
    def _start_stimulations_route() -> Tuple[Response, int]:
        try:
            _name: str = request.form.get('experiment_name', '').strip()
            if not _name:
                _name = 'Unnamed_Experiment'
            _name = _name.replace('/', '_').replace('\\', '_')
                
            _delay_str: str = request.form.get('delay_ms', '0')
            _delay: int = int(_delay_str)

            _loop_str: str = request.form.get('loop_count', '1')
            _loop_count: int = max(1, int(_loop_str))

            _config = _ExperienceConfig(name=_name, delay_ms=_delay, loop_count=_loop_count)
            service._schedule_experience(_config)
            
            return jsonify({"status": "Started"}), 200
        except ValueError:
            return jsonify({"error": "Invalid input format"}), 400

    @_app.route('/video_feed')
    def _video_feed() -> Response:
        return Response(service._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @_app.route('/video', methods=['POST'])
    def _video() -> Tuple[Response, int]:
        _action: str = request.form['action']
        _video_name: str = request.form.get('video_name', 'video_default').strip()

        if _action == "start":
            service._start_auto_video(_video_name, "records")
            return jsonify({"message": "Started"}), 200
        elif _action == "stop":
            service._stop_auto_video()
            return jsonify({"message": "Stopped"}), 200

        return jsonify({"error": "Invalid action"}), 400

    @_app.route('/stop_stimulations', methods=['POST'])
    def _stop_stimulations() -> Response:
        state.stop_stimulations = True
        return jsonify({"status": "Stopped"})

    @_app.route('/status')
    def _status() -> Response:
        return jsonify(running=state.is_running)

    @_app.route('/shutdown', methods=['POST'])
    def _shutdown() -> Response:
        def _do_shutdown() -> None:
            time.sleep(0.3)
            os.kill(os.getpid(), signal.SIGINT)
        threading.Thread(target=_do_shutdown, daemon=True).start()
        return jsonify({"status": "Shutting down"})

    return _app

if __name__ == '__main__':
    _state = _HardwareState()
    _state._connect_arduino()
    _service = _StimulationService(_state)
    _service._start_camera_thread()
    _app = _create_app(_service, _state)
    _app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
