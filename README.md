# Negative Fake Companion Technique (NEGFC)

The present model can be described as a hybrid approach that combines the use of the [VIP_HCI](https://vip.readthedocs.io/en/v1.4.0/#) library and [Tensorflow](https://www.tensorflow.org/) in order to find the coordinates and estimated fluxes of potential companions.

Our methodology draws inspiration from the Negative Fake Companion technique, which employs a normalized Point Spread Function (PSF) to emulate the flux emitted by companions. The underlying concept involves identifying a scale factor that, when applied to the PSF, effectively eliminates the companion from the original image, thereby preserving solely the background noise (see image bellow).

<p align="center">
<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/negfc_example.png?raw=true" alt="Image" width="80%"/>
</center>
</p>

The code implemented in our study leverages the VIP framework to perform Principal Component Analysis (PCA) combined with Angular Differential Imaging (ADI), enabling the identification of sources and determining their respective coordinates. 
Specifically, our objective is to detect potential companions exhibiting a brightness level surpassing a pre-defined signal-to-noise ratio threshold.


## Get started
The easiest way to run this repo is via [Docker](https://www.docker.com/). If you have already Docker installed, run the following commands:
```
bash build_container.sh
```
and then 
```
bash run_container.sh <gpu-option>
```
In the context where <gpu-option> can be set to true, it signifies the requirement for GPU visibility within the container. 
If no option is specified, the container will exclusively utilize the CPU for processing.

#### Anaconda or Python Virtual Environment 
If you want to use another virtual enviroment, we recommend you to use: 
```
pip install -r requirements.txt
```
Additionally, it is necessary to install TensorFlow version 2.10:
```
pip install tensorflow==2.10
```
## Data 
[Download data here](https://drive.google.com/drive/folders/1yXbscZa_bq9u65Rf33VT1IggBBSkSU1j?usp=sharing) 📌

## Directory tree 
```
 📂 NEGFC: Root directory
└─── 📂 core
│     │   📜 data.py: Data loader and preprocessing functions
│     │   📜 engine.py: Main script that uses `model.py` and `data.py` to automate and simplify the execution of the pipeline
│     │   📜 model.py: Flux model implementation
│ 
└─── 📂 figures
│ 
└─── 📜 .gitignore: Files to ignore when pushing on GitHub
└─── 📜 README.md: Markdown readme (what you are reading now)    
└─── 📜 Dockerfile: Docker configuration to create container's image 
└─── 📜 build_container.sh: Script to create a container using the Dockerfile configuration
└─── 📜 run_container.sh: Script to run the container created using `build_container.sh`
└─── 📜 requirements.txt: Python dependencies
└─── 📜 train.py: script to run the pipeline from command line
└─── 📑 tutorial.ipynb: A tutorial, demonstrating the functionality of the pipeline.
```

## Example 
```
python train.py --data ./data/dhtau
```
In this case we are loading `dhtau` folder wich inside it should contains: 
- `center_im.fits`: Cube
- `median_unsat.fits`: PSFs
- `rotnth.fits`: Rotation angles

⚠️ It is crucial to maintain the same file names ⚠️


## To-do 🕥
- ~VIP preprocessing~
- ~First guess using Tensorflow~
- Final optimization using Hamiltonian MCMC
