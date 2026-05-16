# ✋ Hand Puzzle Game

> **웹캠을 이용한 실시간 손동작 인식 퍼즐 게임**  
> Python · OpenCV · MediaPipe · Pillow · Pygame

---

## 📸 게임 화면

| 대기 화면 | 게임 플레이 | 클리어 화면 |
|:---------:|:----------:|:----------:|
| 난이도 선택 | 퍼즐 조작 | 랭킹 표시 |

---

## 🎮 조작 방법

| 제스처 | 동작 |
|--------|------|
| ✊ **주먹 쥐기** | 퍼즐 조각 잡기 |
| 🖐 **손가락 펴기** (2개 이상) | 퍼즐 조각 놓기 |
| ✌️ **손가락 2개** 2초 유지 | 2×2 모드 시작 |
| 🤟 **손가락 3개** 2초 유지 | 3×3 모드 시작 |
| 🖖 **손가락 4개** 2초 유지 | 4×4 모드 시작 |
| 🖐 **손 전체 펴기** 3초 유지 | 게임 리셋 |
| 👌 **OK 제스처** 3초 유지 | 프로그램 종료 |

---

## ✨ 주요 기능

- **스냅샷 기반 퍼즐** — 게임 시작 순간 웹캠 화면을 캡처해 정지 이미지로 퍼즐 생성
- **난이도 선택** — 손가락 개수로 2×2 / 3×3 / 4×4 선택
- **무조건 격자 스냅** — 조각을 놓으면 가장 가까운 칸으로 자동 흡착
- **조각 교환** — 다른 조각이 있는 칸에 놓으면 두 조각의 위치가 교환됨
- **Z-Order 관리** — 잡은 조각이 항상 최상위 레이어에 표시
- **타이머 & 이동 횟수** — 실제로 교환이 일어난 경우만 이동 횟수 카운트
- **랭킹 시스템** — 난이도별 TOP 5 기록을 `puzzle_ranking.json`에 저장
- **한글 렌더링** — Pillow + 맑은 고딕으로 한글 텍스트 출력
- **효과음** — 조각 잡기(뾱!) / 스냅(착!) / 클리어(빠바밤~!) 사운드 합성

---

## 🛠 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yjh943677/hand-puzzle-game.git
cd hand-puzzle-game
```

### 2. 가상환경 생성 (권장)
```bash
conda create -n puzzle python=3.10 -y
conda activate puzzle
```

### 3. 패키지 설치
```bash
pip install opencv-python mediapipe numpy pillow pygame
```

### 4. 실행
```bash
python hand_puzzle_game.py
```

> 처음 실행 시 MediaPipe 모델 파일(`hand_landmarker.task`, 약 20MB)이 자동 다운로드됩니다.

---

## 📦 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `opencv-python` | 4.x | 웹캠 캡처 및 화면 렌더링 |
| `mediapipe` | 0.10.x | 손 랜드마크 인식 (Tasks API) |
| `numpy` | 1.x | 이미지 배열 처리 및 사운드 합성 |
| `pillow` | 10.x | 한글 텍스트 렌더링 (맑은 고딕) |
| `pygame` | 2.x | 효과음 재생 |

---

## 📁 파일 구조

```
hand-puzzle-game/
├── hand_puzzle_game.py     # 메인 게임 코드
├── hand_landmarker.task    # MediaPipe 모델 (자동 다운로드)
├── puzzle_ranking.json     # 랭킹 저장 파일 (자동 생성)
└── README.md
```

---

## 🏗 클래스 구조

```
SoundManager          효과음 합성 및 재생
KoreanTextRenderer    Pillow 기반 한글 텍스트 렌더링
PuzzlePiece           퍼즐 조각 상태 관리
HandDetector          MediaPipe Tasks API 래퍼 (손 인식)
PuzzleGame            게임 전체 로직 및 메인 루프
```

---

## 🎯 게임 흐름

```
[WAITING] 대기 화면
    │  손가락 2/3/4개 2초 유지
    ▼
[PLAYING] 게임 진행
    │  주먹=잡기, 펴기=놓기, 손 전체 펴기 3초=리셋
    ▼
[CLEAR] 클리어 화면 + 랭킹 표시
    │  손 전체 펴기 3초 → 다시 WAITING
```

---

## 📝 개발 환경

- **OS**: Windows 10/11
- **Python**: 3.10
- **웹캠**: 내장 카메라 또는 외장 USB 웹캠

---

## 👩‍💻 개발자

- [yjh943677](https://github.com/yjh943677)
