The MNIST folder consists the files for the implementation of MNIST digit classifier using CNN on the pynq z2 board.

This project focuses on developing a hardware design for digit classification from image data to deploy on PYNQ Z2. Convolutional Neural Network is the most efficient method for image classification. A very fundamental CNN Architecture is developed, trained and fined tuned on the MNIST (Modified National Institute of Standards and Technology) digit classification dataset using keras from Tensorflow.

The trained model is then translated into hardware design using VITIS High Level Synthesis (HLS). The model is replicated using cpp for HLS. A custom IP core is synthesized by the VITIS HLS 2023.2. It is then integrated with the ZYNQ Processing system using VIVADO 2023.2 and then the bitstream file is generated from the block design in VIVADO.

The PYNQ Z2 board is accessed through the jupyter notebook and the custom overlay is dumped over Programmable Logic (PL) using PYNQ library of python. The test images are sent to the PL through the M_AXI interface and the performance of the design is verified.
