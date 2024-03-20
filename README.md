# Emuru
Emul (エムル, Emuru) is a vorpal rabbit also the guide for the unique quest that Sunraku encounters, titled “Invitation from Rabituza”, and a main NPC character. Emul is a great ally with powerful magic, like the ability to teleport other characters.

## Installation
```bash
git clone https://github.com/aimagelab/Emuru.git && cd Emuru
```

```bash
conda create --name teddy python==3.11.5
conda activate teddy
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python -m nltk.downloader all
```

Download the files necessary to run the code:
```bash
wget -qO- https://github.com/aimagelab/Emuru/releases/download/download/font_square.tar.gz | tar xvz -C files
```
