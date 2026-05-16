# ✋ Hand Puzzle Game

> **웹캠을 이용한 실시간 손동작 인식 퍼즐 게임**  
> Python · OpenCV · MediaPipe · Pillow · Pygame

---

## 🎮 조작 방법

| 제스처 | 동작 |
|--------|------|
| ✊ **주먹 쥐기** | 퍼즐 조각 잡기 |
| 🖐 **손가락 펴기** (2개 이상) | 퍼즐 조각 놓기 |
| ✌️ **손가락 2개** 2초 유지 | 2×2 모드 시작 |
| 🤟 **손가락 3개** 2초 유지 | 3×3 모드 시작 |
| 🖖 **손가락 4개** 2초 유지 | 4×4 모드 시작 |
| 🖐 **손 전체 펴기** (5개) 3초 유지 | 게임 리셋 |
| 👌 **OK 제스처** 3초 유지 | 프로그램 종료 |

> 난이도 선택 시 **엄지 제외** 손가락 수로 인식  
> 리셋·종료 제스처는 게임 도중 실수 방지를 위해 **3초 유지** 필요

---

## ✨ 주요 기능

### 🧩 퍼즐
- **스냅샷 기반** — 게임 시작 순간 웹캠 화면을 캡처해 정지 이미지로 퍼즐 생성 (조각 내부가 움직이지 않음)
- **난이도 선택** — 손가락 개수로 2×2 / 3×3 / 4×4 선택
- **조각 교환** — 다른 조각이 있는 칸에 놓으면 두 조각의 위치가 자동 교환
- **무조건 격자 스냅** — 조각을 놓으면 항상 가장 가까운 칸으로 자동 흡착 (허공 부유 없음)

### 🔒 정답 조각 잠금
- 정답 자리에 놓인 조각은 **파란 테두리 + 모서리 잠금 표시**로 구분
- 잠긴 조각은 **집을 수 없고**, 다른 조각도 그 자리에 **밀어넣을 수 없음**
- 교환으로 인해 간접적으로 정답이 된 조각도 자동으로 잠금 처리

### 📊 점수 & 랭킹
- **타이머** — 첫 조각을 잡는 순간부터 시작, 클리어 시 고정
- **이동 횟수** — 실제로 두 조각이 교환된 경우만 카운트 (집었다 제자리에 놓으면 미카운트)
- **랭킹 시스템** — 난이도별 TOP 5 기록을 `puzzle_ranking.json`에 저장 (이동 횟수 적은 순 → 시간 짧은 순)

### 🔊 효과음
- 조각 잡기: **뾱!** (팝 사운드)
- 조각 스냅: **착!** (자석 사운드)
- 게임 클리어: **빠바밤~!** (팡파르)
- 외부 파일 없이 numpy로 직접 합성

### 🖥️ 기술
- **OpenCL 가속** — Intel GPU (UHD Graphics) 활용
- **MediaPipe Tasks API** — Hand Landmarker 모델로 21개 관절 실시간 추출
- **mcp 기반 손가락 판정** — 카메라 각도에 무관한 안정적 인식
- **맑은 고딕 한글** — Pillow ImageDraw로 한글 텍스트 렌더링

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

> 처음 실행 시 MediaPipe 모델 파일(`hand_landmarker.task`, 약 20MB)이 바탕화면에 자동 다운로드됩니다.

---

## 📦 의존성

| 패키지 | 용도 |
|--------|------|
| `opencv-python` | 웹캠 캡처, 화면 렌더링, OpenCL 가속 |
| `mediapipe` | 손 랜드마크 인식 (Tasks API) |
| `numpy` | 이미지 배열 처리 및 효과음 합성 |
| `pillow` | 한글 텍스트 렌더링 (맑은 고딕) |
| `pygame` | 효과음 재생 |

---

## 📁 파일 구조

```
hand-puzzle-game/
├── hand_puzzle_game.py     # 메인 게임 코드
├── hand_landmarker.task    # MediaPipe 모델 (실행 시 자동 다운로드)
├── puzzle_ranking.json     # 랭킹 저장 파일 (자동 생성)
├── .gitignore
└── README.md
```

---

## 🏗 클래스 구조

| 클래스 | 역할 |
|--------|------|
| `SoundManager` | numpy 기반 효과음 합성 및 재생 |
| `KoreanTextRenderer` | Pillow 기반 한글 텍스트 렌더링 |
| `PuzzlePiece` | 퍼즐 조각 상태 관리 (위치, 잠금, 드래그) |
| `HandDetector` | MediaPipe Tasks API 래퍼 (손 인식, 제스처 판정) |
| `PuzzleGame` | 게임 전체 로직 및 메인 루프 |

---

## 🎯 게임 흐름

```
[WAITING] 대기 화면
    │  손가락 2/3/4개 2초 유지 → 스냅샷 캡처 & 퍼즐 생성
    ▼
[PLAYING] 게임 진행
    │  주먹=잡기, 손펼치기=놓기
    │  조각 교환 → 정답 자리 자동 잠금
    │  손 전체 펴기 3초 → 리셋 (WAITING으로)
    ▼
[CLEAR] 클리어 화면
    │  기록 표시 + 난이도별 랭킹 TOP5
    │  손 전체 펴기 3초 → 재시작 (WAITING으로)
```

---

## 🤖 사용 모델

**MediaPipe Hand Landmarker** (Google)
- 2단계 파이프라인: Palm Detection → Hand Landmark Detection
- 단일 프레임에서 손의 **21개 관절 좌표** 실시간 추출
- 추론: **CPU (XNNPACK delegate)** + **OpenCL (Intel UHD Graphics)**

---

## 📝 개발 환경

- **OS**: Windows 10/11
- **Python**: 3.10
- **GPU**: Intel UHD Graphics 630 (OpenCL)
- **웹캠**: 내장 카메라 또는 외장 USB 웹캠

---

## 👩‍💻 개발자

- [yjh943677](https://github.com/yjh943677)
