# PUResNet(Predicting protein-ligand binding sites using deep convolutional neural network)
Prediction of protein-ligand binding site is fundamental step in understanding functional characteristics of the protein which plays vital role in carrying out different biological functions and is a crucial stage in drug discovery. A protein shows its true nature after interacting with any capable molecule knows as ligand which binds only in favorable binding site of protein structure.
# Requirements
1. Tensorflow 1.11 (https://www.tensorflow.org/)
2. Keras (https://keras.io/)
3. Scipy (https://www.scipy.org/)
4. Scikit-Image (https://scikit-image.org/)
5. Open Babel (http://openbabel.org/wiki/Main_Page)
6. Pybel (http://openbabel.org/docs/current/UseTheLibrary/Python_Pybel.html)
7. TFBIO (https://gitlab.com/cheminfIBB/tfbio)
8. Numpy (https://numpy.org/)
9. Python 3.6 (https://www.python.org/)<br>
Note that: It is better to setup new environment using conda or pyenv. You may need to compile open babel and tfbio if installing with PIP doesn't work.
# Model Architecture
<img src="M1.jpg" style="float: left; margin-right: 10px;"/>
<h5 align="center"> Figure showing Convolutional Block,Identiy Block and Up Sampling Block </h5>
<br>
<img src="M2.jpg" style="float: left; margin-right: 10px;"/>
<h5 align="center"> Figure showing Architecture of PUResNet </h5>
<h1>Usage</h1>
1. Clone this repository 
<pre>
git clone https://github.com/jivankandel/PUResNet.git
cd PUResNet
</pre>
2. Setup Environment
<pre>
#create conda environment
conda create -n env_name python=3.6 
conda activate env_name
</pre>
