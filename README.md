# Vision Transformer Memory as a Differentiable Search Index for Image Retrieval

## 📚 Project Overview

This project explores a novel [information retrieval (IR)](https://en.wikipedia.org/wiki/Information_retrieval) framework applied to image retrieval that utilizes a **differentiable function** to generate a **sorted list of image identifiers** in response to a given **image query**.

The approach is called **Differentiable Search Index (DSI)**, and was originally proposed in the paper [Transformer Memory as a Differentiable Search Index](https://arxiv.org/pdf/2202.06991.pdf) by researchers at Google Research.

In its original formulation, **DSI** aims at both encompassing all document's corpus information and executing retrieval within a single **Transformer language model**, instead of adopting the index-then-retrieve pipeline used in most modern IR sytems.

The notebook file **"vision_transformer_dsi.ipynb"** presents the implemented DSI solution applied to an image retrieval task: a **Sequence to Sequence Vision Transformer** (ViT) model `f` that, given an image query `q` as input, returns a list of image IDs ranked by relevance to the given image query, and compares its performance with a traditional "index-then-retrieve" approach based on a **BoVW** baseline model.

We evaluate the performance of the proposed models using the **Indexing Accuracy**, **Mean Average Precision (MAP)** and **Recall at K** metrics computed on multiple variations of the **ImageNet** and the **MS COCO** datasets, and we compare the results obtained for multiple ViT variations and  configurations with the aforementioned **BoVW** baseline.

## 📝 Author

**Valerio Di Stefano** - _"Sapienza" University of Rome_
<br/>
Email: [distefano.1898728@studenti.uniroma1.it](mailto:distefano.1898728@studenti.uniroma1.it)

## 🔗 External Links

* **Main Related Work**: [Transformer Memory as a Differentiable Search Index](https://arxiv.org/pdf/2202.06991.pdf)

  _Authors_: Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, Donald Metzler
  
* **Project Repository**: [GitHub Repository](https://github.com/valeriodiste/computer_vision_project)

* **Project Presentation**: [Google Slides](https://docs.google.com/presentation/d/1RPvnGxorEW1WhZ6iUhBGnqFb-Wj6cVxzn3LFu01M6qI/edit?usp=sharing)

* **Project Notebook**: [vision_transformer_dsi.ipynb](https://drive.google.com/file/d/1xqJit0FAr_XR67uxtqCTeaNph37rkAPe/view?usp=sharing)

