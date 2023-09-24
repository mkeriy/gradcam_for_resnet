# gradcam_for_resnet
## Installation Guide
To get a local working copy of the development repository, do:
```shell
git clone https://github.com/mkeriy/gradcam_for_resnet.git
cd gradcam_for_resnet
```
Create virtual environment:
```shell
pip install virtualenv
python -m venv <virtual-env-name>
source <virtual-env-name>/bin/activate
```
Upload the required libraries:
```shell
pip install -r requirements.txt
```
If you have problems with torch installation, try:
```shell
pip install torch --no-cache-dir
pip install -r requirements.txt
```
