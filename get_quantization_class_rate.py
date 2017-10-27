#!/usr/bin/python3
import subprocess, shlex
import glob, os
from tqdm import tqdm
import threading
import multiprocessing.dummy as mp 


class AtomicCounter:
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

def runProcess(exe):    
    p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(True):
      retcode = p.poll() #returns None while subprocess is running
      line = p.stdout.readline()
      yield line
      if(retcode is not None):
        break
        

def classifyAndCheck(image_path, debug=False):

    global graph
    global config2016
    global top
    global groundTruth
    global Total_Correct_Predictions
   
    config = config2016
    img = image_path.split("/")[-1]
    
    folder = image_path.split(img)[0]
    predicted_name = classify(image_path, debug=debug)
    pred = check(folder,img, config2016, groundTruth, predicted_name, debug=debug)
    if pred:
        Total_Correct_Predictions.increment()

def classify(image_path,  debug=True):

#    graph, config, top,
    global graph
    global config2016
    global top
    
    config = config2016

    cmd = tool + ' --graph='+graph+' --input_width='+config["width"]+' --input_height='+config["height"]+' --input_mean='+config["mean"]+' --input_std='+config["mean"]+' --input_layer="'+config["i_label"]+'" --output_layer="'+config["o_label"]+'" --image="'+image_path+'" --labels="'+config["labels"]+'"'
    args = shlex.split(cmd)
    if debug:
        print("Classifying: ", image_path)
    
    res = None
    
    if top == 1:
        top_class = None  
        for idx, line in enumerate(runProcess(args)):
            if idx == 1:
                top_class = line.decode("utf-8")
        res = top_class.split(" ")[-3]
    else:
        # top 5
        top_classes = []
        for idx, line in enumerate(runProcess(args)):
            if debug:
                print(line)
            if idx < 6 and idx > 0:
                A = line.decode("utf-8")
                top_classes.append(A.split(" ")[-3])
        res = top_classes
        
    
    return res

def check(image_dir, image_file, config, groundTruth, predicted_name, debug=False):
    label_file = config["labels"]
   
    fp = open(groundTruth)
    lines = fp.readlines()
    fp.close()

    label = image_dir.split("/")[-2]
    
    if debug:
        print("Ground Truth", label)
        print("Predicted", predicted_name)
    for line, x in enumerate(lines):
        if type(predicted_name) is not list:
            predicted_name = [predicted_name]

        for idx, name in enumerate(predicted_name):
            if name in x:
                suspect_line= lines[line]
                if label in suspect_line:
                    if debug:
                        print("Prediction #",idx,"Found at", line, lines[line])
                    return True

    return False
            


DEBUG     = False
THREADING = False
QUANTIZED = False
TOP_N     = 1
NUM_THREADS = 32 #32
SUB_FOLDERS = True

tool    = 'Tools/tensorflow/examples/label_image/label_image'
if QUANTIZED:
    graph   = "usb/Scripts/TensorflowQuantization/inception_v3_2016_08_28_frozen_QUANTIZED.pb"
else:
    graph   = 'usb/Scripts/TensorflowQuantization/inception_v3_2016_08_28_frozen.pb'

top = TOP_N

config2016 = {
    "width"  : str(299),
    "height" : str(299),
    "mean"   : str(128),
    "std"    : str(128),
    "i_label": "input:0",
    "o_label": "InceptionV3/Predictions/Reshape_1:0",
    "labels" : "usb/Scripts/TensorflowQuantization/imagenet_slim_labels.txt"
}

groundTruth = "usb/Scripts/TensorflowQuantization/ImageNetClasses.txt"


# Datset Images
# image_dir  = "/home/ian/Personal/Masters/Datasets/Custom/"
image_dir = "/tmp/usb/ILSVRC2012_img_val/" #n01440764/
image_file = "goldfish.jpg"

cv_img = []

Total_Images = 0

#Shite Code

progress_total=0
image_paths = []
for f_idx, dr in enumerate(os.walk(image_dir)):
    if SUB_FOLDERS:
        for d in dr[1]:
            folder = dr[0] + d +"/"
            for x_x, x in enumerate(os.walk(folder)):               
                for i_idx, img in enumerate(x[2]):
                    image_paths.append(folder+img)
                    progress_total += 1
                    Total_Images +=1            
    else:
        folder = str(dr[0]) 
        for i_idx, img in enumerate(dr[2]):
            image_paths.append(folder+img)
            progress_total += 1
            Total_Images +=1
            
if THREADING:
    Total_Correct_Predictions = AtomicCounter()  
    pool=mp.Pool(NUM_THREADS)    
    for _ in tqdm(pool.imap_unordered(classifyAndCheck, image_paths), total=len(image_paths)):
        pass

    Total_Images = len(image_paths)
    print("Total Images Checked: ", Total_Images)
    print("Total Correct Predictions", Total_Correct_Predictions.value)
    print("Prediction Rate:", (Total_Correct_Predictions.value/Total_Images) * 100)

else:
    Total_Correct_Predictions = 0
    with tqdm(total=progress_total) as pbar:
        for f_idx, dr in enumerate(os.walk(image_dir)):
            folder = str(dr[0]) + "/"
            for i_idx, img in enumerate(dr[2]):
               Total_Images +=1
               image_file = img
               predicted_name = classify(folder+img, debug=DEBUG)
               #pred = check(folder,img, config2016, groundTruth, predicted_name, debug=DEBUG)
               #if pred:
               #    Total_Correct_Predictions += 1
               pbar.update(1)
    pbar.close()
               
    print("Total Images Checked: ", Total_Images)
    print("Total Correct Predictions", Total_Correct_Predictions)
    print("Prediction Rate:", (Total_Correct_Predictions/Total_Images) * 100)


