# OscarNet
Applying image segmentation techniques on the popular Spot Garbage dataset



install cv2
install torch -- conda install pytorch torchvision -c pytorch

install fvcore -- pip install git+https://github.com/facebookresearch/fvcore.git

install detectron2 -- 

install pycocotools -- git clone git@github.com:cocodataset/cocoapi.git
                        make # might be not necessary
                        python setup.py build
                        python setup.py install
      
      
      
### A few results from our trained model
<img src="https://github.com/rathoreanirudh/OscarNet/blob/master/data/results/bb_can.jpg" height="480" width="480">
<img src="https://github.com/rathoreanirudh/OscarNet/blob/master/data/results/bb_red_trash.jpg" height="480" width="480">
<img src="https://github.com/rathoreanirudh/OscarNet/blob/master/data/results/bb_chips.jpg" height="480" width="620">
<img src="https://github.com/rathoreanirudh/OscarNet/blob/master/data/results/hunteroutput.png" height="480" width="620">
<img src="https://github.com/rathoreanirudh/OscarNet/blob/master/data/results/splash_chips.jpg" height="480" width="620">
