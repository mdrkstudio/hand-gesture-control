from __future__ import annotations

import argparse
from collections import deque
import ctypes
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def load_dependencies() -> None:
    """Load vision dependencies only when starting the controller."""
    global cv2, mp
    cache = Path.home() / "Library" / "Caches" / "mdrk-hand-gesture-control" / "matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "Missing vision dependencies. Use Python 3.11 and install requirements.txt "
            "in a fresh virtual environment."
        ) from error
    require_hands_api()


WINDOW_TITLE = "Hand Gesture Control"
OPEN_PALM_MIN_EXTENDED_FINGERS = 3
THUMB_DIRECTION_RATIO = 1.35
THUMB_MIN_LENGTH = 0.42
LEFT_THUMB_OUTER_LEFT_THRESHOLD = 0.30
THUMB_HOLD_SECONDS = 0.45
THUMB_ACTION_COOLDOWN_SECONDS = 0.85
THUMB_SCROLL_COOLDOWN_SECONDS = 0.45
PLAY_PAUSE_HOLD_SECONDS = 0.7
PLAY_PAUSE_COOLDOWN_SECONDS = 1.2
PALM_HOLD_MOVE_TOLERANCE = 0.035
VOLUME_MIN_DISTANCE = 0.16
VOLUME_MAX_DISTANCE = 1.35
VOLUME_ENTER_DISTANCE = 0.34
VOLUME_ENTER_HOLD_SECONDS = 0.22
VOLUME_SMOOTHING = 0.22
VOLUME_UPDATE_SECONDS = 0.05
VOLUME_UPDATE_STEP = 1
DEFAULT_BACKGROUND_FPS = 24.0
STATUS_PRINT_INTERVAL_SECONDS = 0.25


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


class AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


class CoreAudioVolume:
    SYSTEM_OBJECT = 1
    MAIN_ELEMENT = 0
    GLOBAL_SCOPE = fourcc("glob")
    OUTPUT_SCOPE = fourcc("outp")
    DEFAULT_OUTPUT_DEVICE = fourcc("dOut")
    VOLUME_SCALAR = fourcc("volm")
    VIRTUAL_MASTER_VOLUME = fourcc("vmas")

    def __init__(self) -> None:
        self._lib = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )
        self._lib.AudioObjectHasProperty.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(AudioObjectPropertyAddress),
        ]
        self._lib.AudioObjectHasProperty.restype = ctypes.c_ubyte
        self._lib.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(AudioObjectPropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        self._lib.AudioObjectGetPropertyData.restype = ctypes.c_int32
        self._lib.AudioObjectSetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(AudioObjectPropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._lib.AudioObjectSetPropertyData.restype = ctypes.c_int32

    def _address(
        self, selector: int, scope: int, element: int
    ) -> AudioObjectPropertyAddress:
        return AudioObjectPropertyAddress(selector, scope, element)

    def _has_property(
        self, object_id: int, address: AudioObjectPropertyAddress
    ) -> bool:
        return bool(self._lib.AudioObjectHasProperty(object_id, ctypes.byref(address)))

    def _get_float(
        self, object_id: int, address: AudioObjectPropertyAddress
    ) -> float | None:
        if not self._has_property(object_id, address):
            return None

        value = ctypes.c_float()
        size = ctypes.c_uint32(ctypes.sizeof(value))
        status = self._lib.AudioObjectGetPropertyData(
            object_id,
            ctypes.byref(address),
            0,
            None,
            ctypes.byref(size),
            ctypes.byref(value),
        )
        if status != 0:
            return None
        return float(value.value)

    def _set_float(
        self, object_id: int, address: AudioObjectPropertyAddress, value: float
    ) -> bool:
        if not self._has_property(object_id, address):
            return False

        scalar = ctypes.c_float(float(clamp(value, 0.0, 1.0)))
        status = self._lib.AudioObjectSetPropertyData(
            object_id,
            ctypes.byref(address),
            0,
            None,
            ctypes.sizeof(scalar),
            ctypes.byref(scalar),
        )
        return status == 0

    def _default_output_device(self) -> int | None:
        address = self._address(
            self.DEFAULT_OUTPUT_DEVICE, self.GLOBAL_SCOPE, self.MAIN_ELEMENT
        )
        device = ctypes.c_uint32()
        size = ctypes.c_uint32(ctypes.sizeof(device))
        status = self._lib.AudioObjectGetPropertyData(
            self.SYSTEM_OBJECT,
            ctypes.byref(address),
            0,
            None,
            ctypes.byref(size),
            ctypes.byref(device),
        )
        if status != 0 or device.value == 0:
            return None
        return int(device.value)

    def get_percent(self) -> int:
        device = self._default_output_device()
        if device is None:
            return 50

        for selector in (self.VIRTUAL_MASTER_VOLUME, self.VOLUME_SCALAR):
            for element in (self.MAIN_ELEMENT, 1, 2):
                value = self._get_float(
                    device, self._address(selector, self.OUTPUT_SCOPE, element)
                )
                if value is not None:
                    return int(round(clamp(value, 0.0, 1.0) * 100))

        return 50

    def set_percent(self, percent: float) -> bool:
        device = self._default_output_device()
        if device is None:
            return False

        scalar = clamp(percent, 0, 100) / 100.0
        for selector in (self.VIRTUAL_MASTER_VOLUME, self.VOLUME_SCALAR):
            master_address = self._address(
                selector, self.OUTPUT_SCOPE, self.MAIN_ELEMENT
            )
            if self._set_float(device, master_address, scalar):
                return True

            changed = False
            for element in (1, 2):
                address = self._address(selector, self.OUTPUT_SCOPE, element)
                changed = self._set_float(device, address, scalar) or changed
            if changed:
                return True

        return False


class GestureController:
    def __init__(self, swap_hands: bool = False) -> None:
        self.swap_hands = swap_hands
        self.volume = CoreAudioVolume()
        self.volume_active = False
        self.volume_enter_started_at: float | None = None
        self.current_volume = self.volume.get_percent()
        self.smoothed_volume = float(self.current_volume)
        self.last_volume_update = 0.0
        self.thumb_candidate_direction = {"Left": None, "Right": None}
        self.thumb_candidate_started_at = {"Left": 0.0, "Right": 0.0}
        self.last_right_thumb_action = 0.0
        self.last_left_thumb_action = 0.0
        self.palm_hold_started_at: float | None = None
        self.palm_hold_origin: tuple[float, float] | None = None
        self.last_play_pause_time = 0.0
        self.status_text = "Ready"
        self.status_until = 0.0

    def process(self, results) -> None:
        now = time.monotonic()
        hands = self._hands_by_label(results)
        right_hand = hands.get("Right")
        left_hand = hands.get("Left")

        self._process_volume(right_hand, now)
        self._process_play_pause(right_hand, now)
        self._process_right_thumb_media(right_hand, now)
        self._process_left_thumb_controls(left_hand, now)

    def status(self) -> str:
        if time.monotonic() > self.status_until:
            return "Ready"
        return self.status_text

    def _set_status(self, text: str, seconds: float = 1.2) -> None:
        self.status_text = text
        self.status_until = time.monotonic() + seconds

    def _hands_by_label(self, results) -> dict[str, object]:
        hands: dict[str, object] = {}
        if not results.multi_hand_landmarks:
            return hands

        handedness = results.multi_handedness or []
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            if index < len(handedness) and handedness[index].classification:
                label = handedness[index].classification[0].label
            if self.swap_hands:
                label = {"Left": "Right", "Right": "Left"}.get(label, label)
            hands.setdefault(label, landmarks)

        return hands

    def _process_volume(self, landmarks, now: float) -> None:
        if landmarks is None or not is_volume_ready_pose(landmarks):
            if self.volume_active:
                self._set_status(f"Volume {self.current_volume}%")
            self.volume_active = False
            self.volume_enter_started_at = None
            return

        if not self.volume_active:
            if not is_volume_enter_pose(landmarks):
                self.volume_enter_started_at = None
                return
            if self.volume_enter_started_at is None:
                self.volume_enter_started_at = now
                self._set_status("Hold pinch for volume", 0.25)
                return
            if now - self.volume_enter_started_at < VOLUME_ENTER_HOLD_SECONDS:
                return

            self.volume_active = True
            self.volume_enter_started_at = None
            self.smoothed_volume = float(self.current_volume)
            self._set_status(f"Volume {self.current_volume}%")

        target_volume = volume_from_thumb_index_distance(landmarks)

        self.smoothed_volume += (target_volume - self.smoothed_volume) * VOLUME_SMOOTHING
        next_volume = int(round(clamp(self.smoothed_volume, 0, 100)))
        if (
            now - self.last_volume_update >= VOLUME_UPDATE_SECONDS
            and abs(next_volume - self.current_volume) >= VOLUME_UPDATE_STEP
        ):
            if self.volume.set_percent(next_volume):
                self.current_volume = next_volume
                self.last_volume_update = now
                self._set_status(f"Volume {self.current_volume}%", 0.4)
            else:
                self._set_status("No audio output", 1.5)

    def _process_play_pause(self, landmarks, now: float) -> None:
        if landmarks is None or self.volume_active or not is_full_open_palm(landmarks):
            self.palm_hold_started_at = None
            self.palm_hold_origin = None
            return

        center = palm_center(landmarks)
        if self.palm_hold_started_at is None or self.palm_hold_origin is None:
            self.palm_hold_started_at = now
            self.palm_hold_origin = center
            return

        if point_distance(center, self.palm_hold_origin) > PALM_HOLD_MOVE_TOLERANCE:
            self.palm_hold_started_at = now
            self.palm_hold_origin = center
            return

        if now - self.last_play_pause_time < PLAY_PAUSE_COOLDOWN_SECONDS:
            return
        if now - self.palm_hold_started_at < PLAY_PAUSE_HOLD_SECONDS:
            self._set_status("Hold palm for play/pause", 0.25)
            return

        if press_media_key("play_pause"):
            self._set_status("Play / pause")
        else:
            self._set_status("Open Music/Spotify or Accessibility")

        self.last_play_pause_time = now
        self.palm_hold_started_at = None
        self.palm_hold_origin = None

    def _process_right_thumb_media(self, landmarks, now: float) -> None:
        direction = self._held_thumb_direction("Right", landmarks, now)
        if direction not in ("left", "right"):
            return
        if now - self.last_right_thumb_action < THUMB_ACTION_COOLDOWN_SECONDS:
            return

        action = "next" if direction == "right" else "previous"
        if press_media_key(action):
            self._set_status("Next track" if action == "next" else "Previous track")
        else:
            self._set_status("Open Music/Spotify or Accessibility")

        self.last_right_thumb_action = now
        self._reset_thumb_candidate("Right")

    def _process_left_thumb_controls(self, landmarks, now: float) -> None:
        direction = self._held_thumb_direction("Left", landmarks, now)
        if direction is None:
            return
        cooldown = (
            THUMB_SCROLL_COOLDOWN_SECONDS
            if direction in ("up", "down")
            else THUMB_ACTION_COOLDOWN_SECONDS
        )
        if now - self.last_left_thumb_action < cooldown:
            return

        if direction in ("up", "down"):
            if page_scroll(direction):
                self._set_status("Page up" if direction == "up" else "Page down", 0.4)
            else:
                self._set_status("Enable Accessibility")
        else:
            if switch_desktop(direction):
                arrow = "right" if direction == "right" else "left"
                self._set_status(f"Desktop {arrow}")
            else:
                self._set_status("Enable Accessibility")

        self.last_left_thumb_action = now
        self._reset_thumb_candidate("Left")

    def _held_thumb_direction(
        self, hand_label: str, landmarks, now: float
    ) -> str | None:
        if landmarks is None or self.volume_active:
            self._reset_thumb_candidate(hand_label)
            return None

        direction = thumb_direction(landmarks, hand_label=hand_label)
        if direction is None:
            self._reset_thumb_candidate(hand_label)
            return None

        if self.thumb_candidate_direction[hand_label] != direction:
            self.thumb_candidate_direction[hand_label] = direction
            self.thumb_candidate_started_at[hand_label] = now
            self._set_status(f"Hold {hand_label.lower()} thumb {direction}", 0.3)
            return None

        if now - self.thumb_candidate_started_at[hand_label] < THUMB_HOLD_SECONDS:
            self._set_status(f"Hold {hand_label.lower()} thumb {direction}", 0.3)
            return None

        return direction

    def _reset_thumb_candidate(self, hand_label: str) -> None:
        self.thumb_candidate_direction[hand_label] = None
        self.thumb_candidate_started_at[hand_label] = 0.0


def switch_desktop(direction: str) -> bool:
    key_code = 124 if direction == "right" else 123
    return send_key_code(key_code, modifier="control down")


def page_scroll(direction: str) -> bool:
    key_code = 116 if direction == "up" else 121
    return send_key_code(key_code)


def press_media_key(action: str) -> bool:
    if send_media_app_command(action):
        return True
    if send_media_remote_command(action):
        return True

    key_codes = {
        "previous": 98,
        "play_pause": 100,
        "next": 101,
    }
    key_code = key_codes[action]
    return send_key_code(key_code)


def send_media_remote_command(action: str) -> bool:
    commands = {
        "play_pause": 2,
        "next": 4,
        "previous": 5,
    }
    try:
        media_remote = ctypes.cdll.LoadLibrary(
            "/System/Library/PrivateFrameworks/MediaRemote.framework/MediaRemote"
        )
        media_remote.MRMediaRemoteSendCommand.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        media_remote.MRMediaRemoteSendCommand.restype = None
        media_remote.MRMediaRemoteSendCommand(commands[action], None)
    except (AttributeError, OSError):
        return False

    return True


def send_media_app_command(action: str) -> bool:
    commands = {
        "previous": "previous track",
        "play_pause": "playpause",
        "next": "next track",
    }
    command = commands[action]
    script = f"""
set handled to false
try
    if application "Spotify" is running then
        tell application "Spotify" to {command}
        set handled to true
    end if
end try
try
    if not handled and application "Music" is running then
        tell application "Music" to {command}
        set handled to true
    end if
end try
return handled
"""
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return completed.returncode == 0 and completed.stdout.strip().lower() == "true"


def send_key_code(key_code: int, modifier: str | None = None) -> bool:
    script = f'tell application "System Events" to key code {key_code}'
    if modifier:
        script += f" using {modifier}"

    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return completed.returncode == 0


def landmark_distance(first, second) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


def point_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def hand_diagonal(landmarks) -> float:
    xs = [landmark.x for landmark in landmarks.landmark]
    ys = [landmark.y for landmark in landmarks.landmark]
    return max(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5, 0.001)


def palm_scale(landmarks) -> float:
    return max(landmark_distance(landmarks.landmark[0], landmarks.landmark[9]), 0.001)


def thumb_index_distance(landmarks) -> float:
    distance = landmark_distance(landmarks.landmark[4], landmarks.landmark[8])
    return distance / palm_scale(landmarks)


def is_volume_ready_pose(landmarks) -> bool:
    fingers = extended_fingers(landmarks)
    return not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]


def is_volume_enter_pose(landmarks) -> bool:
    return (
        is_volume_ready_pose(landmarks)
        and thumb_index_distance(landmarks) <= VOLUME_ENTER_DISTANCE
    )


def volume_from_thumb_index_distance(landmarks) -> int:
    distance = thumb_index_distance(landmarks)
    normalized = (distance - VOLUME_MIN_DISTANCE) / (
        VOLUME_MAX_DISTANCE - VOLUME_MIN_DISTANCE
    )
    return int(round(clamp(normalized, 0.0, 1.0) * 100))


def finger_is_extended(landmarks, tip: int, pip: int) -> bool:
    return landmarks.landmark[tip].y < landmarks.landmark[pip].y - 0.025


def extended_fingers(landmarks) -> dict[str, bool]:
    return {
        "index": finger_is_extended(landmarks, 8, 6),
        "middle": finger_is_extended(landmarks, 12, 10),
        "ring": finger_is_extended(landmarks, 16, 14),
        "pinky": finger_is_extended(landmarks, 20, 18),
    }


def is_full_open_palm(landmarks) -> bool:
    return all(extended_fingers(landmarks).values())


def is_open_palm(landmarks) -> bool:
    extended = sum(extended_fingers(landmarks).values())
    return extended >= OPEN_PALM_MIN_EXTENDED_FINGERS


def is_fist_without_thumb(landmarks) -> bool:
    return not any(extended_fingers(landmarks).values())


def thumb_direction(landmarks, hand_label: str | None = None) -> str | None:
    if not is_fist_without_thumb(landmarks):
        return None

    thumb_base = landmarks.landmark[2]
    thumb_tip = landmarks.landmark[4]
    scale = palm_scale(landmarks)
    dx = (thumb_tip.x - thumb_base.x) / scale
    dy = (thumb_tip.y - thumb_base.y) / scale
    length = (dx * dx + dy * dy) ** 0.5

    if length < THUMB_MIN_LENGTH:
        return None

    if hand_label == "Left":
        wrist = landmarks.landmark[0]
        index_base = landmarks.landmark[5]
        pinky_base = landmarks.landmark[17]
        palm_left_edge = min(wrist.x, index_base.x, pinky_base.x)
        outer_left = (palm_left_edge - thumb_tip.x) / scale
        if outer_left > LEFT_THUMB_OUTER_LEFT_THRESHOLD and abs(dx) > abs(dy) * 0.75:
            return "left"

    if abs(dx) > abs(dy) * THUMB_DIRECTION_RATIO:
        return "right" if dx > 0 else "left"
    if abs(dy) > abs(dx) * THUMB_DIRECTION_RATIO:
        return "down" if dy > 0 else "up"
    return None


def is_two_finger_pose(landmarks) -> bool:
    fingers = extended_fingers(landmarks)
    return (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    )


def palm_center(landmarks) -> tuple[float, float]:
    indexes = (0, 5, 9, 13, 17)
    x = sum(landmarks.landmark[index].x for index in indexes) / len(indexes)
    y = sum(landmarks.landmark[index].y for index in indexes) / len(indexes)
    return x, y


def two_finger_center(landmarks) -> tuple[float, float]:
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    return (index_tip.x + middle_tip.x) / 2.0, (index_tip.y + middle_tip.y) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe hand tracking demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum hands to track.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run gestures in the background without an OpenCV preview window.",
    )
    parser.add_argument(
        "--background-fps",
        type=float,
        default=DEFAULT_BACKGROUND_FPS,
        help="Frame rate limit for --no-preview. Use 0 to disable the limit.",
    )
    parser.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe left/right labels if your camera reports them backwards.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show hand tracking without sending any system or media controls.",
    )
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    if args.max_hands not in (1, 2):
        parser.error("--max-hands must be 1 or 2")
    if not math.isfinite(args.background_fps) or args.background_fps < 0:
        parser.error("--background-fps must be a finite nonnegative number")
    if args.preview_only and args.no_preview:
        parser.error("--preview-only cannot be combined with --no-preview")
    return args


def require_hands_api():
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError(
            "This MediaPipe build has no mp.solutions.hands. "
            "Install the pinned MediaPipe version from requirements.txt in a fresh Python 3.11 environment."
        )

    return (
        mp.solutions.hands,
        mp.solutions.drawing_utils,
        mp.solutions.drawing_styles,
    )


def create_hands(max_hands: int, static_image_mode: bool = False):
    mp_hands, _, _ = require_hands_api()
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_hands,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def open_camera(camera_index: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()

    raise RuntimeError(
        f"Could not open camera {camera_index}. "
        "Check macOS camera permission for Terminal/Codex/your IDE."
    )


def draw_results(frame, results) -> int:
    mp_hands, mp_drawing, mp_styles = require_hands_api()

    if not results.multi_hand_landmarks:
        return 0

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

    return len(results.multi_hand_landmarks)


def draw_overlay(frame, hands_count: int, gesture_status: str) -> None:
    lines = [
        f"Hands: {hands_count}",
        "Right thumb+index pinch, then spread: volume",
        "Right palm hold: play/pause",
        "Right fist + thumb left/right: track",
        "Left fist + thumb up/down: page",
        "Left fist + thumb left/right: desktop",
        gesture_status,
    ]

    y = 34
    for index, line in enumerate(lines):
        color = (0, 255, 0) if index in (0, len(lines) - 1) else (230, 230, 230)
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72 if index else 0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30


class StatusPrinter:
    def __init__(self) -> None:
        self.last_status = ""
        self.last_printed_at = 0.0

    def update(self, status: str) -> None:
        now = time.monotonic()
        if status == self.last_status:
            return
        if now - self.last_printed_at < STATUS_PRINT_INTERVAL_SECONDS:
            return

        print(f"[gesture] {status}", flush=True)
        self.last_status = status
        self.last_printed_at = now


def run_camera(
    camera_index: int,
    max_hands: int,
    swap_hands: bool,
    preview: bool,
    background_fps: float,
    preview_only: bool = False,
) -> None:
    cap = open_camera(camera_index)
    status_printer = StatusPrinter()
    min_frame_seconds = 1.0 / background_fps if not preview and background_fps > 0 else 0.0

    if not preview:
        print(
            "Gesture control is running in the background. Press Ctrl+C to stop.",
            flush=True,
        )

    try:
        gestures = None if preview_only else GestureController(swap_hands=swap_hands)
        with create_hands(max_hands=max_hands) as hands:
            while True:
                frame_started_at = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the camera.")

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                if gestures is None:
                    status = "Preview only - system controls disabled"
                else:
                    gestures.process(results)
                    status = gestures.status()

                if preview:
                    hands_count = draw_results(frame, results)
                    draw_overlay(frame, hands_count, status)
                    cv2.imshow(WINDOW_TITLE, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                else:
                    status_printer.update(status)
                    elapsed = time.monotonic() - frame_started_at
                    if elapsed < min_frame_seconds:
                        time.sleep(min_frame_seconds - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    if sys.platform != "darwin":
        print("Hand Gesture Control requires macOS.", file=sys.stderr)
        return 2
    try:
        load_dependencies()
        run_camera(
            args.camera,
            args.max_hands,
            args.swap_hands,
            preview=not args.no_preview,
            background_fps=args.background_fps,
            preview_only=args.preview_only,
        )
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError) as error:
        print(f"Could not start Hand Gesture Control: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
