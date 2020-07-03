### Code for the paper: Misc-GAN: A Multi-scale Generative Model for Graphs

### Requirement:
* Python, Matlab
* Python package: tensorflow=1.1
* Other version of tensorflow might work, but it might disable the gpu.

### Environment and Installation:
1. cd MRCGAN_network/amg-master
2. matlab -nodisplay -nosplash -nodesktop -r mexall
3. conda env create -f environment.yml
4. conda activate miscgan

### Command
1. Training:
python main_network.py --demo --stage training --gpu

2. Testing and evaluation:
python main_network.py --demo --stage testing --gpu


### Some important parameters:
* --demo: Demo
* --dataset_A: the path of the graph in domain A. Domain A usually refers to the original graph.
* --dataset_B: the path of the graph in domain B. Domain B usually refers to the synthetic graph.
* --epoch: the number of epochs
* --Starting_layer: Considering the memory limitation, we start the training process at the second or third layer to reduce the run time and memory. The value should be in the range of 1-4.
* --stage: training stage or testing stage
* --iter: Iterations of residule_block function for generator in the training stage.
* --gpu: Use GPU to train the model or not. The default value is False.
* --clusters: the number of clusters. Here, we cluster the nodes in the graph into 2 group
* --which_direction: domain A to domain B or domain B to domain A.
* --kernel_number: number of kernels in the initial convolutional neural network


### Evaluation：
There are two evaluation methods shown in the testing stage.
1. Plot the generated graph.
2. KL divergence of graph degree.

### Note:
The final results will be displayed in the terminal and the graphs are saved at './graph'.
The performance increases as we set the parameter 'starting_layer' to be 1 or 2. However, when we set the parameter 'starting_layer' to be 3 or 4, the runtime reduces rapidly. There is a trade-off between performance and the runtime.


### Reference:
@article{zhou2019misc, <br/>
  title={Misc-GAN: A Multi-scale Generative Model for Graphs}, <br/>
  author={Zhou, Dawei and Zheng, Lecheng and Xu, Jiejun and He, Jingrui}, <br/>
  journal={Frontiers in Big Data}, <br/>
  volume={2}, <br/>
  pages={3}, <br/>
  year={2019}, <br/>
  publisher={Frontiers} <br/>
}



