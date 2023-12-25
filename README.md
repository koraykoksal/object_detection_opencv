

## Getting Started
<u> Step 1: Download required files </u> <br>

```
 git clone https://github.com/VirajVaitha123/object-detetection-using-opencv.git
```

- coco.names - The model has been pretrained to detect certain objects. The .names file contains the names of all detectable objects. Your model will detect an integer and then find the relevant object from the coco.names files

- frozen_inference_graph.pb - Weights file. The model was trained on a large dataset. During this process, weights for the machine learning model have been tuned and stored within this file. If you wanted to detect new objects, there weights would be changed when you retrain the model. This is generally a large file and you shouldn't be able to read anything when you open it

- ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. This file is a config file that contains the relevant parameters for opencv to run the model
<br>
<u> Step 2: Open Anaconda terminal and create the conda (virtual) environment <br>
- run the following command relative to your directory to create the environment with the relevant dependencies <br>

```
conda env create -f objectdetectioncv.yml 
```
<br>
<br>
<u> Step 3: Navigate to the source folder and run the main_webcam.py file <br>

```
cd <your-local-filepath>\object-detetection-using-opencv\source
python main_webcam.py
```

