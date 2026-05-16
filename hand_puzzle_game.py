"""
Hand Puzzle Game v7
────────────────────────────────────────────────────────
· MediaPipe Tasks API (binarypb 불필요)
· 잡기/놓기: 핀치 거리 대신 '주먹(손가락 접힘) / 손펼침' 으로 판단
  - 검지~소지 4개 모두 접힘 → 잡기 (주먹)
  - 검지~소지 중 2개 이상 펴짐 → 놓기
  → 손 떨림에 영향 없고 확실하게 구분됨
· 스냅샷 기반 정지 이미지 퍼즐
· 손가락 개수(2/3/4개) 2초 유지 → 난이도 선택
· 놓을 때 가장 가까운 빈 칸으로 무조건 스냅
· 손가락 4개 3초 유지 → 리셋
· Pillow 맑은 고딕 한글 렌더링

설치: pip install opencv-python mediapipe numpy pillow
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

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ══════════════════════════════════════════════════════
#  효과음 합성 (외부 파일 없이 numpy로 직접 생성)
# ══════════════════════════════════════════════════════
class SoundManager:
    def __init__(self):
        self.enabled = PYGAME_OK
        if not self.enabled:
            return
        self._grab   = self._make_grab()
        self._snap   = self._make_snap()
        self._clear  = self._make_clear()

    # ── 뾱! : 짧고 가벼운 팝 사운드 ─────────────
    def _make_grab(self):
        sr, dur = 44100, 0.08
        t = np.linspace(0, dur, int(sr * dur), False)
        # 주파수가 빠르게 떨어지는 팝
        freq = 800 * np.exp(-t * 30)
        wave = np.sin(2 * np.pi * freq * t)
        env  = np.exp(-t * 40)
        return self._to_sound(wave * env)

    # ── 착! : 경쾌한 자석 스냅 ───────────────────
    def _make_snap(self):
        sr, dur = 44100, 0.12
        t = np.linspace(0, dur, int(sr * dur), False)
        # 두 개의 톤이 빠르게 올라가며 착 달라붙는 느낌
        wave  = (np.sin(2 * np.pi * 600 * t) * 0.5 +
                 np.sin(2 * np.pi * 1200 * t) * 0.5)
        env   = np.exp(-t * 25) * (1 - np.exp(-t * 200))
        return self._to_sound(wave * env)

    # ── 빠바밤~! : 승리 팡파르 (3음 상승) ────────
    def _make_clear(self):
        sr   = 44100
        notes = [(523, 0.15), (659, 0.15), (784, 0.15), (1047, 0.35)]
        frames = []
        for freq, dur in notes:
            t    = np.linspace(0, dur, int(sr * dur), False)
            wave = (np.sin(2 * np.pi * freq * t) * 0.6 +
                    np.sin(2 * np.pi * freq * 2 * t) * 0.3 +
                    np.sin(2 * np.pi * freq * 3 * t) * 0.1)
            env  = np.exp(-t * 3) * (1 - np.exp(-t * 80))
            frames.append(wave * env)
        full = np.concatenate(frames)
        return self._to_sound(full)

    def _to_sound(self, wave):
        arr = (wave * 32767).astype(np.int16)
        # 스테레오: 동일 채널을 두 개로 복제
        stereo = np.column_stack([arr, arr])
        return pygame.sndarray.make_sound(stereo)

    def play_grab(self):
        if self.enabled: self._grab.play()

    def play_snap(self):
        if self.enabled: self._snap.play()

    def play_clear(self):
        if self.enabled: self._clear.play()

# ══════════════════════════════════════════════════════
#  OpenCL (Intel GPU) 가속 설정
# ══════════════════════════════════════════════════════
cv2.ocl.setUseOpenCL(True)
if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
    print("[INFO] OpenCL (GPU 가속) 활성화됨")
else:
    print("[INFO] OpenCL 미지원 환경 - CPU로 실행")

# ══════════════════════════════════════════════════════
#  모델 경로
# ══════════════════════════════════════════════════════
_USERPROFILE = os.environ.get("USERPROFILE", os.path.expanduser("~"))
MODEL_PATH   = os.path.join(_USERPROFILE, "Desktop", "hand_landmarker.task")
MODEL_URL  = (
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
        self.font_path = next(
            (p for p in paths if os.path.exists(p)), None)
        self._cache = {}

    def _font(self, size):
        if size not in self._cache:
            self._cache[size] = (
                ImageFont.truetype(self.font_path, size)
                if self.font_path else ImageFont.load_default()
            )
        return self._cache[size]

    def draw(self, frame, text, pos, size=30, color=(255,255,255)):
        img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(img).text(
            pos, text, font=self._font(size),
            fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_centered(self, frame, text, cx, cy, size=30, color=(255,255,255)):
        font = self._font(size)
        bbox = ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox(
            (0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        return self.draw(frame, text, (cx-tw//2, cy-th//2), size, color)


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
    offset_x:    int  = 0
    offset_y:    int  = 0

    def center(self):
        x, y = self.current_pos
        return (x + self.size//2, y + self.size//2)

    def is_inside(self, px, py):
        x, y = self.current_pos
        return x <= px <= x+self.size and y <= py <= y+self.size

    def draw(self, frame, fw, fh):
        x, y = self.current_pos
        s    = self.size
        x1, y1 = max(x,0),   max(y,0)
        x2, y2 = min(x+s,fw), min(y+s,fh)
        tw, th = x2-x1, y2-y1
        if tw<=0 or th<=0: return
        ox, oy = x1-x, y1-y
        tile = self.image[oy:oy+th, ox:ox+tw]
        if tile.size == 0: return

        if self.dragging:
            dst = frame[y1:y2, x1:x2]
            cv2.addWeighted(tile, 0.75, dst, 0.25, 0, dst)
            frame[y1:y2, x1:x2] = dst
        else:
            frame[y1:y2, x1:x2] = tile

        color = (0,255,80) if self.dragging else (200,200,200)
        thick = 3         if self.dragging else 1
        cv2.rectangle(frame, (x1,y1), (x2-1,y2-1), color, thick)

        if self.dragging:
            l, t, c = 18, 3, (0,255,80)
            for (cx,cy),(dx,dy) in zip(
                [(x1,y1),(x2-1,y1),(x1,y2-1),(x2-1,y2-1)],
                [(1,1),  (-1,1),   (1,-1),    (-1,-1)]
            ):
                cv2.line(frame,(cx,cy),(cx+dx*l,cy),c,t)
                cv2.line(frame,(cx,cy),(cx,cy+dy*l),c,t)


# ══════════════════════════════════════════════════════
#  손 인식 (Tasks API)
# ══════════════════════════════════════════════════════
class HandDetector:
    def __init__(self):
        # 한글 경로 문제 우회: 파일을 바이트로 읽어 메모리에서 로드
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
        pts = [(int(lm.x*fw), int(lm.y*fh)) for lm in lms]
        for a,b in HAND_CONN:
            cv2.line(frame, pts[a], pts[b], (0,210,255), 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0,210,255), -1)

    def get_index_tip(self, lms, fw, fh):
        """검지 끝(8번) 픽셀 좌표."""
        return int(lms[8].x*fw), int(lms[8].y*fh)

    def count_extended_fingers(self, lms):
        """tip이 mcp보다 손목에서 60% 이상 멀면 펴진 것 (각도 무관, 양손 호환)."""
        wx, wy = lms[0].x, lms[0].y
        count  = 0
        # 엄지: tip(4) vs mcp(2)
        if (math.hypot(lms[4].x-wx, lms[4].y-wy) >
                math.hypot(lms[2].x-wx, lms[2].y-wy) * 1.3):
            count += 1
        # 검지~소지: tip vs mcp
        for t, m in zip([8,12,16,20], [5,9,13,17]):
            if (math.hypot(lms[t].x-wx, lms[t].y-wy) >
                    math.hypot(lms[m].x-wx, lms[m].y-wy) * 1.6):
                count += 1
        return count

    def count_fingers_no_thumb(self, lms):
        """엄지 제외 검지~소지, mcp 기반 (난이도 선택 전용)."""
        wx, wy = lms[0].x, lms[0].y
        count  = 0
        for t, m in zip([8,12,16,20], [5,9,13,17]):
            if (math.hypot(lms[t].x-wx, lms[t].y-wy) >
                    math.hypot(lms[m].x-wx, lms[m].y-wy) * 1.6):
                count += 1
        return count

    def is_fist(self, lms):
        """주먹: 검지~소지 4개 모두 접힘."""
        wx, wy = lms[0].x, lms[0].y
        for t, m in zip([8,12,16,20], [5,9,13,17]):
            if (math.hypot(lms[t].x-wx, lms[t].y-wy) >
                    math.hypot(lms[m].x-wx, lms[m].y-wy) * 1.6):
                return False   # 하나라도 펴지면 주먹 아님
        return True

    def is_open(self, lms):
        """놓기: 검지~소지 중 2개 이상 펴지면."""
        wx, wy = lms[0].x, lms[0].y
        count  = sum(
            1 for t, m in zip([8,12,16,20], [5,9,13,17])
            if math.hypot(lms[t].x-wx, lms[t].y-wy) >
               math.hypot(lms[m].x-wx, lms[m].y-wy) * 1.6
        )
        return count >= 2

    def is_ok_gesture(self, lms, fw, fh):
        """OK 제스처: 엄지+검지 끝 가깝고, 중지·약지·소지 펴짐, 검지 접힘."""
        wx, wy = lms[0].x, lms[0].y
        # 엄지-검지 픽셀 거리
        pinch = math.hypot((lms[4].x-lms[8].x)*fw,
                            (lms[4].y-lms[8].y)*fh)
        # 중지·약지·소지 펴짐
        others = sum(
            1 for t, m in zip([12,16,20], [9,13,17])
            if math.hypot(lms[t].x-wx, lms[t].y-wy) >
               math.hypot(lms[m].x-wx, lms[m].y-wy) * 1.6
        ) >= 2
        # 검지 접힘 (tip이 mcp보다 멀지 않음)
        index_bent = not (
            math.hypot(lms[8].x-wx, lms[8].y-wy) >
            math.hypot(lms[5].x-wx, lms[5].y-wy) * 1.6
        )
        return pinch < 55 and others and index_bent

    def close(self):
        self._lm.close()


# ══════════════════════════════════════════════════════
#  게임
# ══════════════════════════════════════════════════════
class PuzzleGame:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.W, self.H = 1280, 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.state    = GameState.WAITING
        self.renderer = KoreanTextRenderer()
        self.detector = HandDetector()
        self.sound    = SoundManager()

        self.PUZZLE_PX = 540
        self.board_x   = (self.W - self.PUZZLE_PX) // 2
        self.board_y   = (self.H - self.PUZZLE_PX) // 2

        self.pieces: list[PuzzlePiece] = []
        self.grid_size  = 3
        self.selected   = None
        self.grabbed    = False   # True = 주먹 상태 (잡기 중)

        self.move_count = 0
        self.start_time = None
        self.clear_elapsed = 0   # 클리어 시점 경과시간 고정값

        self.sel_value  = None
        self.sel_start  = None
        self.reset_start  = None
        self.flash_alpha  = 0
        self.grabbed_from = None  # 조각을 잡기 전 원래 슬롯 위치
        self.quit_start   = None  # OK 제스처 종료 타이머

    # ── 퍼즐 생성 ─────────────────────────────
    def create_puzzle(self, snapshot, grid_size):
        self.grid_size = grid_size
        tile_px = self.PUZZLE_PX // grid_size

        # 스냅샷 중앙에서 정사각형으로 크롭 (비율 왜곡 없음)
        h, w = snapshot.shape[:2]
        side  = min(h, w)
        x0    = (w - side) // 2
        y0    = (h - side) // 2
        cropped   = snapshot[y0:y0+side, x0:x0+side]
        board_img = cv2.resize(cropped, (self.PUZZLE_PX, self.PUZZLE_PX))

        self.pieces.clear()
        positions = []
        for row in range(grid_size):
            for col in range(grid_size):
                x = col * tile_px
                y = row * tile_px
                tile    = board_img[y:y+tile_px, x:x+tile_px].copy()
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

        self.move_count = 0
        self.start_time = time.time()
        self.selected   = None
        self.grabbed    = False
        self.state      = GameState.PLAYING

    # ── 랭킹 저장 / 로드 ──────────────────────
    RANK_FILE = "puzzle_ranking.json"
    MAX_RANKS = 5   # 난이도별 상위 5개만 보관

    def _save_ranking(self):
        import json
        key = f"{self.grid_size}x{self.grid_size}"
        record = {"time": self.clear_elapsed, "moves": self.move_count}

        try:
            with open(self.RANK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        if key not in data:
            data[key] = []

        data[key].append(record)
        # 이동 횟수 적은 순 우선, 같으면 시간 짧은 순
        data[key].sort(key=lambda r: (r["moves"], r["time"]))
        data[key] = data[key][:self.MAX_RANKS]

        with open(self.RANK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_ranking(self):
        import json
        key = f"{self.grid_size}x{self.grid_size}"
        try:
            with open(self.RANK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(key, [])
        except Exception:
            return []

    # ── 클리어 체크 ───────────────────────────
    def check_clear(self):
        return all(
            p.current_pos[0] == p.correct_pos[0] and
            p.current_pos[1] == p.correct_pos[1]
            for p in self.pieces
        )

    # ── 가장 가까운 빈 슬롯 ───────────────────
    def find_nearest_grid(self, piece):
        tile_px  = piece.size
        occupied = {p.current_pos for p in self.pieces
                    if p is not piece and not p.dragging}
        best, best_d = piece.correct_pos, float('inf')
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                gx = self.board_x + col * tile_px
                gy = self.board_y + row * tile_px
                if (gx, gy) in occupied:
                    continue
                cx = gx + tile_px // 2
                cy = gy + tile_px // 2
                d  = math.hypot(piece.center()[0] - cx,
                                piece.center()[1] - cy)
                if d < best_d:
                    best_d, best = d, (gx, gy)
        return best

    # ── 놓기 처리 (자리 교환 방식) ───────────
    def _do_drop(self):
        if self.selected is None:
            return

        p       = self.selected
        tile_px = p.size

        # 가장 가까운 슬롯 찾기 (점유 무관)
        best_pos, best_d = p.correct_pos, float('inf')
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                gx = self.board_x + col * tile_px
                gy = self.board_y + row * tile_px
                cx = gx + tile_px // 2
                cy = gy + tile_px // 2
                d  = math.hypot(p.center()[0] - cx, p.center()[1] - cy)
                if d < best_d:
                    best_d, best_pos = d, (gx, gy)

        # 해당 슬롯에 다른 조각 있으면 → grabbed_from(원래 슬롯)으로 교환
        other = next(
            (q for q in self.pieces
             if q is not p and q.current_pos == best_pos), None)
        if other is not None:
            other.current_pos = self.grabbed_from  # 정확한 원래 슬롯으로 이동

        p.current_pos     = best_pos
        p.dragging        = False
        self.selected     = None

        # ── 이동 횟수: 실제로 다른 조각과 자리가 바뀐 경우만 +1 ──
        # other가 있고 (교환 발생) AND 목적지가 원래 자리와 다를 때
        actually_swapped = (other is not None and best_pos != self.grabbed_from)
        self.grabbed_from = None

        if actually_swapped:
            self.move_count += 1
            self.sound.play_snap()   # 착!

        if self.check_clear():
            self.clear_elapsed = int(time.time() - self.start_time)
            self.state = GameState.CLEAR
            self._save_ranking()
            self.sound.play_clear()  # 빠바밤~!

    # ── 퍼즐 인터랙션 ─────────────────────────
    def update_interaction(self, lms):
        if lms is None:
            if self.selected:
                self._do_drop()
                self.grabbed = False
            return

        ix, iy = self.detector.get_index_tip(lms, self.W, self.H)
        fist   = self.detector.is_fist(lms)
        open_  = self.detector.is_open(lms)

        # ── 잡기: 주먹 쥐면 ──────────────────
        if fist and not self.grabbed:
            self.grabbed = True
            for piece in reversed(self.pieces):
                if piece.is_inside(ix, iy):
                    self.selected      = piece
                    piece.dragging     = True
                    px, py             = piece.current_pos
                    piece.offset_x     = ix - px
                    piece.offset_y     = iy - py
                    self.grabbed_from  = piece.current_pos  # 원래 슬롯 저장
                    self.pieces = [p for p in self.pieces if p is not piece]
                    self.pieces.append(piece)
                    self.sound.play_grab()   # 뾱!
                    break

        # ── 드래그: 주먹 유지 중 ─────────────
        if self.grabbed and self.selected:
            p = self.selected
            p.current_pos = (ix - p.offset_x, iy - p.offset_y)

        # ── 놓기: 손 펼치면 ──────────────────
        if self.grabbed and open_:
            self.grabbed = False
            self._do_drop()

    # ── WAITING ───────────────────────────────
    def update_waiting(self, frame, raw_frame, lms):
        if lms is None:
            self.sel_value = self.sel_start = None
            return frame

        # 엄지 포함 전체 5개 펴기는 리셋 전용 → 난이도 선택 제외
        all5 = self.detector.count_extended_fingers(lms)
        if all5 == 5:
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
                # 패널 아래 여백에 표시 (겹치지 않게)
                panel_bottom = (self.H - 220) // 2 + 220 + 20
                frame   = self.renderer.draw_centered(
                    frame,
                    f"{n}x{n}  시작까지  {remain:.1f}초",
                    self.W//2, panel_bottom,
                    size=36, color=(0,255,0))
                if elapsed >= 2.0:
                    self.create_puzzle(raw_frame, n)
                    self.sel_value = self.sel_start = None
        else:
            self.sel_value = self.sel_start = None

        return frame

    # ── 리셋 제스처 ───────────────────────────
    def update_reset_gesture(self, frame, lms):
        if lms is None:
            self.reset_start = None
            return frame

        # 리셋: 손가락 5개 모두 펴기 (엄지 포함 전부)
        n = self.detector.count_extended_fingers(lms)
        is_reset_gesture = (n == 5)

        if is_reset_gesture:
            if self.reset_start is None:
                self.reset_start = time.time()
            elapsed = time.time() - self.reset_start
            prog    = min(elapsed / 3.0, 1.0)
            bh = 18
            y0 = self.H - bh
            cv2.rectangle(frame, (0,y0), (self.W,self.H), (30,30,30), -1)
            cv2.rectangle(frame, (0,y0),
                          (int(self.W*prog), self.H), (0,255,120), -1)
            frame = self.renderer.draw(
                frame, f"리셋: {prog*100:.0f}%  (손 전체 펴기 3초 유지)",
                (10, y0+1), size=17, color=(255,255,255))
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

    # ── UI ────────────────────────────────────
    def draw_ui(self, frame):
        if self.state == GameState.WAITING:
            # 반투명 배경 패널
            overlay = frame.copy()
            pw, ph  = 760, 220
            px      = (self.W - pw) // 2
            py      = (self.H - ph) // 2 - 20
            cv2.rectangle(overlay, (px, py), (px+pw, py+ph),
                          (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            cv2.rectangle(frame, (px, py), (px+pw, py+ph),
                          (0, 180, 255), 2)

            frame = self.renderer.draw_centered(
                frame, "손가락을 펼쳐 난이도를 선택하세요",
                self.W//2, py + 45,
                size=36, color=(255, 255, 255))
            frame = self.renderer.draw_centered(
                frame, "2개: 2x2          3개: 3x3          4개: 4x4",
                self.W//2, py + 95,
                size=30, color=(0, 230, 255))
            frame = self.renderer.draw_centered(
                frame, "주먹 = 잡기   /   손가락 펴기 = 놓기   /   손 전체 펴기 3초 = 리셋",
                self.W//2, py + 145,
                size=22, color=(200, 200, 200))
            frame = self.renderer.draw_centered(
                frame, "OK 3초 = 종료",
                self.W//2, py + 182,
                size=22, color=(100, 180, 255))

        elif self.state == GameState.PLAYING:
            elapsed = int(time.time() - self.start_time)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (self.W,55), (15,15,15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame,
                        f"Time: {elapsed//60:02d}:{elapsed%60:02d}",
                        (20,38), cv2.FONT_HERSHEY_DUPLEX,
                        0.9, (0,230,255), 2)
            cv2.putText(frame, f"Moves: {self.move_count}",
                        (230,38), cv2.FONT_HERSHEY_DUPLEX,
                        0.9, (255,200,0), 2)
            cv2.putText(frame,
                        f"Mode: {self.grid_size}x{self.grid_size}",
                        (450,38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (180,255,180), 1)
            frame = self.renderer.draw(
                frame, "손 전체 펴기 3초 = 리셋  |  OK 3초 = 종료",
                (self.W-420, 18), size=20, color=(150,150,150))

        elif self.state == GameState.CLEAR:
            elapsed = self.clear_elapsed
            overlay = frame.copy()
            # 패널을 더 넓게 (랭킹 표시 공간 확보)
            cv2.rectangle(overlay,
                          (self.W//6, self.H//8),
                          (self.W*5//6, self.H*7//8),
                          (10,10,10), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cx, cy = self.W//2, self.H//2

            # CLEAR!
            txt = "CLEAR!"
            (tw,_),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 2.4, 5)
            cv2.putText(frame, txt, (cx-tw//2, self.H//8+65),
                        cv2.FONT_HERSHEY_DUPLEX, 2.4, (0,255,120), 5)

            # 이번 기록
            frame = self.renderer.draw_centered(
                frame, f"걸린 시간:  {elapsed//60:02d}분 {elapsed%60:02d}초",
                cx, self.H//8+115, size=28, color=(255,255,255))
            frame = self.renderer.draw_centered(
                frame, f"이동 횟수:  {self.move_count}번   난이도: {self.grid_size}x{self.grid_size}",
                cx, self.H//8+155, size=26, color=(180,255,180))

            # 구분선
            cv2.line(frame,
                     (self.W//6+30, self.H//8+175),
                     (self.W*5//6-30, self.H//8+175),
                     (80,80,80), 1)

            # 랭킹 타이틀
            frame = self.renderer.draw_centered(
                frame, f"{self.grid_size}x{self.grid_size} 베스트 랭킹 TOP 5",
                cx, self.H//8+205, size=26, color=(255,215,0))

            # 랭킹 목록
            ranks = self._load_ranking()
            rank_labels = ["1위", "2위", "3위", "4위", "5위"]
            for i, r in enumerate(ranks[:5]):
                t = r["time"]
                m = r["moves"]
                is_me = (t == self.clear_elapsed and m == self.move_count)
                color = (0, 255, 120) if is_me else (200, 200, 200)
                me_tag = "  ◀ 현재" if is_me else ""
                text  = f"{rank_labels[i]}  {t//60:02d}분 {t%60:02d}초  {m}번{me_tag}"
                frame = self.renderer.draw_centered(
                    frame, text,
                    cx, self.H//8 + 240 + i * 38,
                    size=24, color=color)

            frame = self.renderer.draw_centered(
                frame, "손 전체 펴기 3초  →  재시작",
                cx, self.H*7//8 - 20, size=21, color=(150,150,150))

        return frame

    # ── 플래시 ────────────────────────────────
    def draw_flash(self, frame):
        if self.flash_alpha > 0:
            overlay = np.full(frame.shape, 255, dtype=np.uint8)
            a = self.flash_alpha / 255.0
            frame = cv2.addWeighted(overlay, a, frame, 1-a, 0)
            self.flash_alpha = max(0, self.flash_alpha - 20)
        return frame

    # ── 메인 루프 ─────────────────────────────
    def run(self):
        print("[INFO] 시작!")
        print("[INFO] 주먹 쥐기 = 잡기  /  손가락 펴기 = 놓기")
        print("[INFO] 손가락 2/3/4개 2초 유지 = 난이도 선택")
        print("[INFO] 검지 1개 3초 유지 = 리셋")
        print("[INFO] OK 제스처(엄지+검지 O) 3초 = 종료")

        while True:
            ret, raw = self.cap.read()
            if not ret:
                break

            raw   = cv2.flip(raw, 1)
            raw   = cv2.resize(raw, (self.W, self.H))
            frame = raw.copy()

            result = self.detector.detect(raw)
            lms    = result.hand_landmarks[0] \
                     if result.hand_landmarks else None

            if lms:
                self.detector.draw_landmarks(frame, lms, self.W, self.H)

                fist  = self.detector.is_fist(lms)
                open_ = self.detector.is_open(lms)
                n_ext = self.detector.count_extended_fingers(lms)
                ok    = self.detector.is_ok_gesture(lms, self.W, self.H)

                # ── OK 제스처 종료 타이머 ──────────────
                if ok:
                    if self.quit_start is None:
                        self.quit_start = time.time()
                    held = time.time() - self.quit_start
                    prog = min(held / 3.0, 1.0)
                    # 종료 진행 바 (상단)
                    bh = 16
                    cv2.rectangle(frame, (0, 0), (self.W, bh), (30,30,30), -1)
                    cv2.rectangle(frame, (0, 0),
                                  (int(self.W * prog), bh), (0,100,255), -1)
                    frame = self.renderer.draw(
                        frame,
                        f"종료: {prog*100:.0f}%  (OK 제스처 3초 유지)",
                        (10, 1), size=15, color=(255,255,255))
                    if held >= 3.0:
                        break   # 메인 루프 탈출 → 종료
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
                for piece in self.pieces:
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
