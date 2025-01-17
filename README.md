## MR.Q: Towards General-Purpose Model-Free Reinforcement Learning
PyTorch implementation of the MR.Q algorithm from [Towards General-Purpose Model-Free Reinforcement Learning]() by Scott Fujimoto, Pierluca D'Oro, Amy Zhang, Yuandong Tian, and Mike Rabbat.

### Installing

Experiments were originally run with Python 3.11, but Python 3.10-3.12 is supported.
```
git clone git@github.com:facebookresearch/MRQ.git
cd MRQ
pip install -r requirements.txt
```

### Usage

Benchmark is designated by a prefix (Gym-, Dmc-, Dmc-visual-, Atari-) followed by the original environment name. A complete list of environments are contained in [MRQ/utils.py](MRQ/utils.py).

Example usage:
```
cd MRQ
python main.py --env Gym-HalfCheetah-v4
python main.py --env Dmc-quadruped-walk
python main.py --env Dmc-visual-walker-walk
python main.py --env Atari-Pong-v5
```

### Code Structure

- Agent and hyperparameters: [MRQ/MRQ.py](MRQ/MRQ.py).
- Architecture: [MRQ/models.py](MRQ/models.py).
- Replay buffer: [MRQ/buffer.py](MRQ/buffer.py).
- Environment preprocesing: [MRQ/env_preprocessing.py](MRQ/env_preprocessing.py). 

### Results

Results are formatted in human-readable .txt files under [/results](results). There is a code snippet in [MRQ/utils.py](MRQ/utils.py#L46) to process the .txt files into a dictionary of arrays. 
