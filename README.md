## Start Train
### Init the machine
```
sudo apt install git
git clone https://github.com/Touki-dev/2048.git
sudo apt install python3
sudo apt install python3-pip -y
cd 2048
sudo apt install python3.11-venv
python3 -m venv env-2048
source env-2048/bin/activate
pip install tensorflow
pip install numpy
```
### Start training
```
python3 train.py
```
