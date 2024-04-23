# hits-rocks

Real-time clod segmentation. This repo uses two models, yolo and U-net. Also there are wrappers for realtime segmentation in video stream.

Data used from [here](https://homeassistant.kotah.ru/local/clodding_train.avi)

## Get started
Clone repo and install the dependencies.
```sh
git clone https://github.com/ssslakter/hits-rocks
pip install -r requirements.txt
```

# Demo
Use this command to run real-time segmentation of the video with clods
```sh
python ./src/demo_yolo.py ./my_clods.avi  -m ./model.pt
```
## How to use CLI
Since this script uses `cv2.imshow` it must be possible to display image somewhere, so if you are using WSL(like I do), then you won't be able to use it. To 

See [this reddit post](https://www.reddit.com/r/bashonubuntuonwindows/comments/sbeccr/displaying_images_with_opencv_on_wsl2/?rdt=61473) to find more detailed explanation on how to display images with opencv on WSL.

# How to train
First you must have the following structure of the folders (which works both for yolo and U-net models). Note that images and labels must have the same names.
```
data/
|-- train/
|   |-- images/
|   |   |-- 1.jpg
|   |   |-- 2.jpg
|   |   |-- ...
|   |-- labels/
|       |-- 1.txt
|       |-- 2.txt
|       |-- ...
|-- valid/
    |-- images/
    |   |-- 1.jpg
    |   |-- ...
    |-- labels/
        |-- 1.txt
        |-- ...
```