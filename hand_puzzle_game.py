"""
Hand Puzzle Game - Final
────────────────────────────────────────────────────────
· MediaPipe Tasks API (hand_landmarker.task)
· 주먹 = 잡기 / 손 펼치기 = 놓기
· 스냅샷 기반 정지 이미지 퍼즐
· 손가락 개수(2/3/4개) 2초 유지 → 난이도 선택
· 조각 교환 방식 스냅 (정답 조각 잠금)
· 타이머 & 이동 횟수 (실제 교환 시만 카운트)
· 난이도별 TOP5 랭킹 저장
· 효과음 (pygame numpy 합성)
· OpenCL GPU 가속
· Pillow 맑은 고딕 한글 렌더링

설치: pip install opencv-python mediapipe numpy pillow pygame
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import random
import time
import math
import urllib.request
import os
import sys
from enum import Enum
from PIL import ImageFont, ImageDraw, Image
from dataclasses import dataclass

# ── pygame 효과음 ───────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ── OpenCL GPU 가속 ─────────────────────────────────────
cv2.ocl.setUseOpenCL(True)
if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
    print("[INFO] OpenCL (GPU 가속) 활성화됨")
else:
    print("[INFO] OpenCL 미지원 - CPU로 실행")

# ── 모델 경로 ───────────────────────────────────────────
_USERPROFILE = os.environ.get("USERPROFILE", os.path.expanduser("~"))
MODEL_PATH   = os.path.join(_USERPROFILE, "Desktop", "hand_landmarker.task")
MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print("[INFO] 모델 다운로드 중... (약 20MB)")
    def _p(b, bs, t):
        print(f"\r  {min(b*bs/t*100,100):.1f}%", end="", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_p)
    print("\n[INFO] 완료!")


# ══════════════════════════════════════════════════════
#  효과음
# ══════════════════════════════════════════════════════
class SoundManager:
    def __init__(self):
        self.enabled = PYGAME_OK
        if not self.enabled:
            return
        self._grab  = self._make_grab()
        self._snap  = self._make_snap()
        self._clear = self._make_clear()

    def _make_grab(self):
        sr, dur = 44100, 0.08
        t    = np.linspace(0, dur, int(sr * dur), False)
        freq = 800 * np.exp(-t * 30)
        wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 40)
        return self._to_sound(wave)

    def _make_snap(self):
        sr, dur = 44100, 0.12
        t    = np.linspace(0, dur, int(sr * dur), False)
        wave = (np.sin(2 * np.pi * 600 * t) * 0.5 +
                np.sin(2 * np.pi * 1200 * t) * 0.5)
        wave *= np.exp(-t * 25) * (1 - np.exp(-t * 200))
        return self._to_sound(wave)

    def _make_clear(self):
        sr     = 44100
        notes  = [(523, 0.15), (659, 0.15), (784, 0.15), (1047, 0.35)]
        frames = []
        for freq, dur in notes:
            t    = np.linspace(0, dur, int(sr * dur), False)
            wave = (np.sin(2 * np.pi * freq * t) * 0.6 +
                    np.sin(2 * np.pi * freq * 2 * t) * 0.3 +
                    np.sin(2 * np.pi * freq * 3 * t) * 0.1)
            wave *= np.exp(-t * 3) * (1 - np.exp(-t * 80))
            frames.append(wave)
        return self._to_sound(np.concatenate(frames))

    def _to_sound(self, wave):
        arr    = (wave * 32767).astype(np.int16)
        stereo = np.column_stack([arr, arr])
        return pygame.sndarray.make_sound(stereo)

    def play_grab(self):
        if self.enabled: self._grab.play()

    def play_snap(self):
        if self.enabled: self._snap.play()

    def play_clear(self):
        if self.enabled: self._clear.play()


# ══════════════════════════════════════════════════════
#  상태
# ══════════════════════════════════════════════════════
class GameState(Enum):
    WAITING = 0
    PLAYING = 1
    CLEAR   = 2


# ══════════════════════════════════════════════════════
#  한글 렌더러
# ══════════════════════════════════════════════════════
class KoreanTextRenderer:
    def __init__(self):
        paths = [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/malgunbd.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ]
        self.font_path = next((p for p in paths if os.path.exists(p)), None)
        self._cache = {}

    def _font(self, size):
        if size not in self._cache:
            self._cache[size] = (
                ImageFont.truetype(self.font_path, size)
                if self.font_path else ImageFont.load_default()
            )
        return self._cache[size]

    def draw(self, frame, text, pos, size=30, color=(255, 255, 255)):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(img).text(
            pos, text, font=self._font(size),
            fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_centered(self, frame, text, cx, cy, size=30, color=(255, 255, 255)):
        font = self._font(size)
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
            (0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return self.draw(frame, text, (cx - tw // 2, cy - th // 2), size, color)


# ══════════════════════════════════════════════════════
#  퍼즐 조각
# ══════════════════════════════════════════════════════
@dataclass
class PuzzlePiece:
    image:       np.ndarray
    correct_pos: tuple
    current_pos: tuple
    size:        int
    dragging:    bool = False
    snapped:     bool = False
    offset_x:    int  = 0
    offset_y:    int  = 0

    def center(self):
        x, y = self.current_pos
        return (x + self.size // 2, y + self.size // 2)

    def is_inside(self, px, py, margin=15):
        """충돌 판정: margin 픽셀만큼 범위를 넓게 잡음 (작은 조각에서도 잘 잡힘)."""
        x, y = self.current_pos
        return (x - margin <= px <= x + self.size + margin and
                y - margin <= py <= y + self.size + margin)

    def is_correct(self):
        return self.current_pos == self.correct_pos

    def draw(self, frame, fw, fh):
        x, y   = self.current_pos
        s      = self.size
        x1, y1 = max(x, 0),      max(y, 0)
        x2, y2 = min(x + s, fw), min(y + s, fh)
        tw, th = x2 - x1, y2 - y1
        if tw <= 0 or th <= 0:
            return
        ox, oy    = x1 - x, y1 - y
        tile      = self.image[oy:oy + th, ox:ox + tw]
        if tile.size == 0:
            return

        if self.dragging:
            dst = frame[y1:y2, x1:x2]
            cv2.addWeighted(tile, 0.75, dst, 0.25, 0, dst)
            frame[y1:y2, x1:x2] = dst
        else:
            frame[y1:y2, x1:x2] = tile

        # 테두리
        if self.snapped:
            color, thick = (255, 150, 0), 3    # 파란색: 정답 고정
        elif self.dragging:
            color, thick = (0, 255, 80),  3    # 초록: 드래그 중
        else:
            color, thick = (200, 200, 200), 1  # 회색: 일반

        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, thick)

        # 정답 고정: 모서리 잠금 표시
        if self.snapped:
            ls = 8
            for (cx, cy) in [(x1, y1), (x2-1, y1), (x1, y2-1), (x2-1, y2-1)]:
                cv2.rectangle(frame,
                              (cx - ls//2, cy - ls//2),
                              (cx + ls//2, cy + ls//2),
                              (255, 150, 0), -1)

        # 드래그: 코너 마크
        if self.dragging:
            l, t, c = 18, 3, (0, 255, 80)
            for (cx, cy), (dx, dy) in zip(
                [(x1,y1),(x2-1,y1),(x1,y2-1),(x2-1,y2-1)],
                [(1,1),  (-1,1),   (1,-1),    (-1,-1)]
            ):
                cv2.line(frame, (cx, cy), (cx + dx*l, cy), c, t)
                cv2.line(frame, (cx, cy), (cx, cy + dy*l), c, t)


# ══════════════════════════════════════════════════════
#  손 인식
# ══════════════════════════════════════════════════════
class HandDetector:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            model_data = f.read()
        base = mp_python.BaseOptions(model_asset_buffer=model_data)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._lm = mp_vision.HandLandmarker.create_from_options(opts)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._lm.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    def draw_landmarks(self, frame, lms, fw, fh):
        pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in lms]
        for a, b in HAND_CONN:
            cv2.line(frame, pts[a], pts[b], (0, 210, 255), 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0, 210, 255), -1)

    def get_grab_point(self, lms, fw, fh):
        """
        잡기 위치: 검지(8) + 중지(12) + 약지(16) + 소지(20) 끝의 평균
        주먹을 쥐면 네 손가락이 모이므로 손 중심에 가까운 좌표가 나옴
        """
        tips = [8, 12, 16, 20]
        x = sum(lms[t].x for t in tips) / len(tips)
        y = sum(lms[t].y for t in tips) / len(tips)
        return int(x * fw), int(y * fh)

    def _dist_from_wrist(self, lms, idx):
        return math.hypot(lms[idx].x - lms[0].x, lms[idx].y - lms[0].y)

    def count_extended_fingers(self, lms):
        """엄지 포함 5개, mcp 기반 거리 비교 (양손 호환)."""
        count = 0
        if self._dist_from_wrist(lms, 4) > self._dist_from_wrist(lms, 2) * 1.3:
            count += 1
        for t, m in zip([8, 12, 16, 20], [5, 9, 13, 17]):
            if self._dist_from_wrist(lms, t) > self._dist_from_wrist(lms, m) * 1.6:
                count += 1
        return count

    def count_fingers_no_thumb(self, lms):
        """엄지 제외 (난이도 선택 전용)."""
        count = 0
        for t, m in zip([8, 12, 16, 20], [5, 9, 13, 17]):
            if self._dist_from_wrist(lms, t) > self._dist_from_wrist(lms, m) * 1.6:
                count += 1
        return count

    def is_fist(self, lms):
        """주먹: 검지~소지 4개 모두 접힘."""
        for t, m in zip([8, 12, 16, 20], [5, 9, 13, 17]):
            if self._dist_from_wrist(lms, t) > self._dist_from_wrist(lms, m) * 1.6:
                return False
        return True

    def is_open(self, lms):
        """놓기: 검지~소지 중 2개 이상 펴짐."""
        count = sum(
            1 for t, m in zip([8, 12, 16, 20], [5, 9, 13, 17])
            if self._dist_from_wrist(lms, t) > self._dist_from_wrist(lms, m) * 1.6
        )
        return count >= 2

    def is_ok_gesture(self, lms, fw, fh):
        """OK: 엄지+검지 붙이고, 중지·약지·소지 펴짐, 검지 접힘."""
        pinch = math.hypot(
            (lms[4].x - lms[8].x) * fw,
            (lms[4].y - lms[8].y) * fh)
        others = sum(
            1 for t, m in zip([12, 16, 20], [9, 13, 17])
            if self._dist_from_wrist(lms, t) > self._dist_from_wrist(lms, m) * 1.6
        ) >= 2
        index_bent = not (
            self._dist_from_wrist(lms, 8) > self._dist_from_wrist(lms, 5) * 1.6)
        return pinch < 55 and others and index_bent

    def close(self):
        self._lm.close()


# ══════════════════════════════════════════════════════
#  게임
# ══════════════════════════════════════════════════════
class PuzzleGame:

    PUZZLE_PX  = 540
    RANK_FILE  = "puzzle_ranking.json"
    MAX_RANKS  = 5

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.W, self.H = 1280, 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.board_x = (self.W - self.PUZZLE_PX) // 2
        self.board_y = (self.H - self.PUZZLE_PX) // 2

        self.state    = GameState.WAITING
        self.renderer = KoreanTextRenderer()
        self.detector = HandDetector()
        self.sound    = SoundManager()

        self.pieces:    list[PuzzlePiece] = []
        self.grid_size  = 3
        self.selected:  PuzzlePiece | None = None
        self.grabbed    = False
        self.grabbed_from = None

        self.move_count   = 0
        self.start_time   = None
        self.clear_elapsed = 0

        self.sel_value   = None
        self.sel_start   = None
        self.reset_start = None
        self.quit_start  = None
        self.flash_alpha = 0

    # ══════════════════════════════════════════
    #  퍼즐 생성
    # ══════════════════════════════════════════
    def create_puzzle(self, snapshot, grid_size):
        self.grid_size = grid_size
        tile_px = self.PUZZLE_PX // grid_size

        h, w  = snapshot.shape[:2]
        side  = min(h, w)
        x0    = (w - side) // 2
        y0    = (h - side) // 2
        board = cv2.resize(snapshot[y0:y0+side, x0:x0+side],
                           (self.PUZZLE_PX, self.PUZZLE_PX))

        self.pieces.clear()
        positions = []
        for row in range(grid_size):
            for col in range(grid_size):
                x, y    = col * tile_px, row * tile_px
                tile    = board[y:y+tile_px, x:x+tile_px].copy()
                correct = (self.board_x + x, self.board_y + y)
                positions.append(correct)
                self.pieces.append(PuzzlePiece(
                    image=tile, correct_pos=correct,
                    current_pos=correct, size=tile_px))

        while True:
            random.shuffle(positions)
            if all(positions[i] != p.correct_pos
                   for i, p in enumerate(self.pieces)):
                break
        for piece, pos in zip(self.pieces, positions):
            piece.current_pos = pos
            piece.snapped     = False
            piece.dragging    = False

        self.move_count = 0
        self.start_time = time.time()
        self.selected   = None
        self.grabbed    = False
        self.state      = GameState.PLAYING

    # ══════════════════════════════════════════
    #  랭킹
    # ══════════════════════════════════════════
    def _save_ranking(self):
        import json
        key    = f"{self.grid_size}x{self.grid_size}"
        record = {"time": self.clear_elapsed, "moves": self.move_count}
        try:
            with open(self.RANK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        data.setdefault(key, [])
        data[key].append(record)
        data[key].sort(key=lambda r: (r["moves"], r["time"]))
        data[key] = data[key][:self.MAX_RANKS]
        with open(self.RANK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_ranking(self):
        import json
        key = f"{self.grid_size}x{self.grid_size}"
        try:
            with open(self.RANK_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get(key, [])
        except Exception:
            return []

    # ══════════════════════════════════════════
    #  클리어 체크
    # ══════════════════════════════════════════
    def check_clear(self):
        return all(p.snapped for p in self.pieces)

    # ══════════════════════════════════════════
    #  놓기 (교환 + 잠금)
    # ══════════════════════════════════════════
    def _do_drop(self):
        if self.selected is None:
            return
        p       = self.selected
        tile_px = p.size

        # 가장 가까운 슬롯 (snapped 점유 슬롯 제외)
        best_pos, best_d = self.grabbed_from, float('inf')
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                gx = self.board_x + col * tile_px
                gy = self.board_y + row * tile_px
                occ = next((q for q in self.pieces
                             if q is not p and q.current_pos == (gx, gy)), None)
                if occ and occ.snapped:
                    continue
                d = math.hypot(p.center()[0] - (gx + tile_px//2),
                               p.center()[1] - (gy + tile_px//2))
                if d < best_d:
                    best_d, best_pos = d, (gx, gy)

        # 교환 대상 (snapped 아닌 것만)
        other = next((q for q in self.pieces
                      if q is not p and q.current_pos == best_pos
                      and not q.snapped), None)
        if other is not None:
            other.current_pos = self.grabbed_from

        p.current_pos = best_pos
        p.dragging    = False
        p.snapped     = p.is_correct()
        self.selected = None

        # 교환으로 간접 정답된 조각도 잠금
        for piece in self.pieces:
            if not piece.snapped and piece.is_correct():
                piece.snapped = True

        actually_swapped = (other is not None
                            and best_pos != self.grabbed_from)
        self.grabbed_from = None

        if actually_swapped:
            self.move_count += 1
            self.sound.play_snap()

        if self.check_clear():
            self.clear_elapsed = int(time.time() - self.start_time)
            self.state = GameState.CLEAR
            self._save_ranking()
            self.sound.play_clear()

    # ══════════════════════════════════════════
    #  인터랙션
    # ══════════════════════════════════════════
    def update_interaction(self, lms):
        if lms is None:
            if self.selected:
                self._do_drop()
                self.grabbed = False
            return

        ix, iy = self.detector.get_grab_point(lms, self.W, self.H)
        fist   = self.detector.is_fist(lms)
        open_  = self.detector.is_open(lms)

        # 잡기
        if fist and not self.grabbed:
            self.grabbed = True
            for piece in reversed(self.pieces):
                if piece.snapped:
                    continue
                if piece.is_inside(ix, iy):
                    self.selected     = piece
                    piece.dragging    = True
                    px, py            = piece.current_pos
                    piece.offset_x    = ix - px
                    piece.offset_y    = iy - py
                    self.grabbed_from = piece.current_pos
                    self.pieces = [p for p in self.pieces if p is not piece]
                    self.pieces.append(piece)
                    self.sound.play_grab()
                    break

        # 드래그
        if self.grabbed and self.selected:
            p = self.selected
            p.current_pos = (ix - p.offset_x, iy - p.offset_y)

        # 놓기
        if self.grabbed and open_:
            self.grabbed = False
            self._do_drop()

    # ══════════════════════════════════════════
    #  WAITING
    # ══════════════════════════════════════════
    def update_waiting(self, frame, raw_frame, lms):
        if lms is None:
            self.sel_value = self.sel_start = None
            return frame

        # 5개 펴기 = 리셋 전용
        if self.detector.count_extended_fingers(lms) == 5:
            self.sel_value = self.sel_start = None
            return frame

        n = self.detector.count_fingers_no_thumb(lms)
        if n in (2, 3, 4):
            if self.sel_value != n:
                self.sel_value = n
                self.sel_start = time.time()
            else:
                elapsed = time.time() - self.sel_start
                remain  = max(0.0, 2.0 - elapsed)
                panel_bottom = (self.H - 220) // 2 + 220 + 20
                frame = self.renderer.draw_centered(
                    frame,
                    f"{n}x{n}  시작까지  {remain:.1f}초",
                    self.W//2, panel_bottom,
                    size=36, color=(0, 255, 0))
                if elapsed >= 2.0:
                    self.create_puzzle(raw_frame, n)
                    self.sel_value = self.sel_start = None
        else:
            self.sel_value = self.sel_start = None
        return frame

    # ══════════════════════════════════════════
    #  리셋 제스처
    # ══════════════════════════════════════════
    def update_reset_gesture(self, frame, lms):
        if lms is None:
            self.reset_start = None
            return frame

        if self.detector.count_extended_fingers(lms) == 5:
            if self.reset_start is None:
                self.reset_start = time.time()
            elapsed = time.time() - self.reset_start
            prog    = min(elapsed / 3.0, 1.0)
            bh = 18
            y0 = self.H - bh
            cv2.rectangle(frame, (0, y0), (self.W, self.H), (30, 30, 30), -1)
            cv2.rectangle(frame, (0, y0),
                          (int(self.W * prog), self.H), (0, 255, 120), -1)
            frame = self.renderer.draw(
                frame, f"리셋: {prog*100:.0f}%  (손 전체 펴기 3초 유지)",
                (10, y0+1), size=17, color=(255, 255, 255))
            if elapsed >= 3.0:
                self.flash_alpha  = 255
                self.state        = GameState.WAITING
                self.pieces.clear()
                self.selected     = None
                self.grabbed      = False
                self.move_count   = 0
                self.reset_start  = None
        else:
            self.reset_start = None
        return frame

    # ══════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════
    def draw_ui(self, frame):
        if self.state == GameState.WAITING:
            overlay = frame.copy()
            pw, ph  = 760, 220
            px      = (self.W - pw) // 2
            py      = (self.H - ph) // 2 - 20
            cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 180, 255), 2)
            frame = self.renderer.draw_centered(
                frame, "손가락을 펼쳐 난이도를 선택하세요",
                self.W//2, py+45, size=36, color=(255, 255, 255))
            frame = self.renderer.draw_centered(
                frame, "2개: 2x2          3개: 3x3          4개: 4x4",
                self.W//2, py+95, size=30, color=(0, 230, 255))
            frame = self.renderer.draw_centered(
                frame, "주먹 = 잡기   /   손가락 펴기 = 놓기   /   손 전체 펴기 3초 = 리셋",
                self.W//2, py+145, size=22, color=(200, 200, 200))
            frame = self.renderer.draw_centered(
                frame, "OK 3초 = 종료",
                self.W//2, py+182, size=22, color=(100, 180, 255))

        elif self.state == GameState.PLAYING:
            elapsed = int(time.time() - self.start_time)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.W, 55), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"Time: {elapsed//60:02d}:{elapsed%60:02d}",
                        (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 230, 255), 2)
            cv2.putText(frame, f"Moves: {self.move_count}",
                        (230, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 200, 0), 2)
            cv2.putText(frame, f"Mode: {self.grid_size}x{self.grid_size}",
                        (450, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 255, 180), 1)
            frame = self.renderer.draw(
                frame, "손 전체 펴기 3초 = 리셋  |  OK 3초 = 종료",
                (self.W-420, 18), size=20, color=(150, 150, 150))

        elif self.state == GameState.CLEAR:
            elapsed = self.clear_elapsed
            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (self.W//6, self.H//8),
                          (self.W*5//6, self.H*7//8),
                          (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cx = self.W // 2

            txt = "CLEAR!"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 2.4, 5)
            cv2.putText(frame, txt, (cx-tw//2, self.H//8+65),
                        cv2.FONT_HERSHEY_DUPLEX, 2.4, (0, 255, 120), 5)
            frame = self.renderer.draw_centered(
                frame, f"걸린 시간:  {elapsed//60:02d}분 {elapsed%60:02d}초",
                cx, self.H//8+115, size=28, color=(255, 255, 255))
            frame = self.renderer.draw_centered(
                frame, f"이동 횟수:  {self.move_count}번   난이도: {self.grid_size}x{self.grid_size}",
                cx, self.H//8+155, size=26, color=(180, 255, 180))
            cv2.line(frame,
                     (self.W//6+30, self.H//8+175),
                     (self.W*5//6-30, self.H//8+175),
                     (80, 80, 80), 1)
            frame = self.renderer.draw_centered(
                frame, f"{self.grid_size}x{self.grid_size} 베스트 랭킹 TOP 5",
                cx, self.H//8+205, size=26, color=(255, 215, 0))

            ranks       = self._load_ranking()
            rank_labels = ["1위", "2위", "3위", "4위", "5위"]
            for i, r in enumerate(ranks[:5]):
                t      = r["time"]
                m      = r["moves"]
                is_me  = (t == self.clear_elapsed and m == self.move_count)
                color  = (0, 255, 120) if is_me else (200, 200, 200)
                me_tag = "  ◀ 현재" if is_me else ""
                frame  = self.renderer.draw_centered(
                    frame,
                    f"{rank_labels[i]}  {t//60:02d}분 {t%60:02d}초  {m}번{me_tag}",
                    cx, self.H//8 + 240 + i * 38,
                    size=24, color=color)
            frame = self.renderer.draw_centered(
                frame, "손 전체 펴기 3초  →  재시작",
                cx, self.H*7//8-20, size=21, color=(150, 150, 150))

        return frame

    # ══════════════════════════════════════════
    #  플래시
    # ══════════════════════════════════════════
    def draw_flash(self, frame):
        if self.flash_alpha > 0:
            overlay = np.full(frame.shape, 255, dtype=np.uint8)
            frame   = cv2.addWeighted(overlay, self.flash_alpha/255.0,
                                      frame, 1 - self.flash_alpha/255.0, 0)
            self.flash_alpha = max(0, self.flash_alpha - 20)
        return frame

    # ══════════════════════════════════════════
    #  메인 루프
    # ══════════════════════════════════════════
    def run(self):
        print("[INFO] 시작!")
        print("[INFO] 주먹=잡기 / 손펼치기=놓기 / 손전체3초=리셋 / OK3초=종료")
        print("[INFO] 손가락 2/3/4개 2초 유지 → 난이도 선택")

        while True:
            ret, raw = self.cap.read()
            if not ret:
                break

            raw   = cv2.flip(raw, 1)
            raw   = cv2.resize(raw, (self.W, self.H))
            frame = raw.copy()

            result = self.detector.detect(raw)
            lms    = result.hand_landmarks[0] if result.hand_landmarks else None

            if lms:
                self.detector.draw_landmarks(frame, lms, self.W, self.H)
                ok = self.detector.is_ok_gesture(lms, self.W, self.H)

                # OK 종료 제스처
                if ok:
                    if self.quit_start is None:
                        self.quit_start = time.time()
                    held = time.time() - self.quit_start
                    prog = min(held / 3.0, 1.0)
                    cv2.rectangle(frame, (0, 0), (self.W, 16), (30, 30, 30), -1)
                    cv2.rectangle(frame, (0, 0),
                                  (int(self.W * prog), 16), (0, 100, 255), -1)
                    frame = self.renderer.draw(
                        frame, f"종료: {prog*100:.0f}%  (OK 제스처 3초 유지)",
                        (10, 1), size=15, color=(255, 255, 255))
                    if held >= 3.0:
                        break
                else:
                    self.quit_start = None
            else:
                self.quit_start = None

            # 상태별 처리
            if self.state == GameState.WAITING:
                frame = self.update_waiting(frame, raw, lms)

            elif self.state == GameState.PLAYING:
                self.update_interaction(lms)
                frame = self.update_reset_gesture(frame, lms)
                # snapped 먼저(아래), 미완성 나중(위)
                for piece in self.pieces:
                    if piece.snapped:
                        piece.draw(frame, self.W, self.H)
                for piece in self.pieces:
                    if not piece.snapped:
                        piece.draw(frame, self.W, self.H)

            elif self.state == GameState.CLEAR:
                for piece in self.pieces:
                    piece.draw(frame, self.W, self.H)
                frame = self.update_reset_gesture(frame, lms)

            frame = self.draw_ui(frame)
            frame = self.draw_flash(frame)

            cv2.imshow("Hand Puzzle Game", frame)
            cv2.waitKey(1)

        self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════
#  실행
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    ensure_model()
    PuzzleGame().run()
