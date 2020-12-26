# MacZero

Reinforcement learning implementations for playing Punch-Out!! for the NES. ROM, Lua script for interface with the game are from [pykitml](https://github.com/RainingComputers/pykitml).

## Algorithms

- LSTM (PyTorch reimplementation of [this pykitml version](https://github.com/RainingComputers/NES-Punchout-AI)

Future

- DQN

## Usage

### Install requirements

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
sudo apt-get install fceux
sudo apt-get install lua5.1-socket
```

### Run

- Change to the appropriate directory
- `python bot.py`
- `fceux`
- Options (Alt-o) > Video Config (v) > Set X and Y scaling factors to 2.0
- File > Open ROM (Ctrl-O) > open `punchout_rom.nes`
- File (Alt-f) > Load Lua Script (u) > open `fceux_client.lua`
- Set the Fceux window to full screen so that the game is in the top left corner
  - If the bot cannot beat Glass Joe, adjust the pixel offset of the top to match your OS. The setting of 224 works for the thickness of the title bars in Windows 10.
