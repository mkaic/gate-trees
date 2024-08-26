
# The idea
What if I used stochastic gradient updates on an XNOR+bitcount binary neural network and trained it to implicitly represent an image?


# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d --gpus all -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```
