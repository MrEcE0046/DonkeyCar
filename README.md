# RC Line Follower RL Project

Detta projekt tränar en RC-bil i simulering för att följa en linje med hjälp av Reinforcement Learning (PPO) och deployar sedan modellen på en Raspberry Pi.

## Projektstruktur

- `sim/`: Simulator byggd i PyBullet/Gymnasium.
- `training/`: RL-träningsloop och CNN-policy definition.
- `export/`: Verktyg för att exportera modeller till ONNX.
- `raspi/`: Programvara för Raspberry Pi (kamera, inferens, PWM).
- `docs/`: Dokumentation och README.

## Installation

### 1. Förbered miljö
Vi rekommenderar Python 3.11 eller 3.12.

```bash
pip install -r requirements.txt
```

*Obs: På Windows kan PyBullet kräva "Microsoft C++ Build Tools" för att installeras korrekt.*

### 2. Träning
För att starta träningen:

```bash
python training/train.py
```

Modeller sparas i katalogen `models/`.

### 3. Export
När du är nöjd med resultatet, konvertera till ONNX:

```bash
python export/export_onnx.py
```

### 4. Raspberry Pi Deployment
Kopiera `raspi/`-mappen och den exporterade `line_follower.onnx` till din Pi.
Kör sedan:

```bash
python raspi/main.py
```

## Krav
- Python 3.11+
- PyBullet
- Gymnasium
- Stable Baselines 3
- ONNX Runtime
- OpenCV
- RPi.GPIO / pigpio (för Raspberry Pi)
