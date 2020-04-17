# CRAFT: Character-Region Awareness For Text detection
An implementation of CRAFT text detector| [Paper](https://arxiv.org/abs/1904.01941) | [Official Pytorch implementation](https://github.com/clovaai/CRAFT-pytorch)

##Overview
Tensorflow implementation for CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.

## Updates
**17 April, 2020**: Initial update

## Getting started
### Install dependencies
#### Requirements
- tensorflow-gpu>=2.1.0
- opencv-contrib-python>=4.2.0.32
- check requirements.txt
```
pip install -r requirements.txt
```

### Training
#### Prepare data set
- [A Large Chinese Text Dataset in the Wild(CTW)](https://ctwdataset.github.io/)
- Run train.py
''' (with python 3.6)
python train.py --real_data_path /PATH/TO/YOUR/DATASET
'''

### Arguments
* "--alpha": weight of loss of foreground
* "--learning_rate": initial learning rate that will decay every 10,000 iterations [[ref.]](https://github.com/clovaai/CRAFT-pytorch/issues/18#issuecomment-513258344)
* "--batch_size": batch size
* "--canvas_size": size of input image will be resized to this size
* "--mag_ratio": image magnification ratio
* "--real_data_path": dataset with ground-true label
* "--iterations": maximum number of training iterations
* "--weight_dir": directory to save model weights
* TODO "--use_fake": weakly supervised learning
* TODO "--gpu_list": list of gpus to use


### Testing
* Run with pretrained model
''' (with python 3.6)
python test.py --test_folder /PATH/TO/TESTING/IMAGES --weight_dir /PATH/TO/YOUR/WEIGHTS
'''

The result image and socre maps will be saved to `./result` by default.

### Arguments
* `--test_folder`: folder path to input images
* `--weight_dir`: pretrained model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--canvas_size`: image size for inference (depend on pre-trained model)
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* TODO `--show_time`: show processing time
* TODO `--refine`: use link refiner for sentence-level dataset
* TODO `--refiner_model`: pre-trained refiner model


## License
```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
