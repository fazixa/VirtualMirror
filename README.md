# Virtual-Mirror

## how to run

- install Dlib
- if you are using ubuntu run the following commands for Dlib to be installed
```
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev 
sudo apt-get install libx11-dev libgtk-3-dev
sudo apt-get install python python-dev python-pip
sudo apt-get install python3 python3-dev python3-pip
pip install numpy
pip install dlib

```
- go to root of the project and intsall the requirements using the commands below
```
pip install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```
- run webcam.py
