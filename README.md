First, a standard machine learning environment with CUDA / CUDNN / OpenCV should be set up. An introduction, using Ubuntu 16.04 as an example, can be found on my [personal website](https://dennisnotes.com/note/ubuntu-16.04-machine-learning-setup/).

# Additional steps for this special project

## Tkinter
Tkinter is required, install it via `sudo apt-get install python-tk python3-tk tk-dev`.

## Pytorch Development Environment
Now it's time to setup the python environment. The name of the environment (in this case - pytorch), may be changed to whatever you want.
```
mkvirtualenv pytorch -p python3 --system-site-packages
workon pytorch
cd ~/.virtualenvs/pytorch/lib/python3.5/site-packages
ln -s /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
pip install torchvision
cd ~/git/ofp_app
pip install -r requirements.txt
cp config_template.py config.py
```
Download the [pretrained OpenPose model](https://cogsys.reutlingen-university.de/pub/files/op_converted.pth). Store this models somewhere, we'll assume `~/data/op_converted.pth`. Set the path in config.py accordingly, e.g. `self.pose_estimator.model_state_file = "~/data/op_converted.pth"`

TODO: Update..

