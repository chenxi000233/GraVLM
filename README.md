# GraVLM: A Hierarchical Graph-Aligned Vision-Language Model for Cross-Modal Retrieval in Remote Sensing

This repository provides the official implementation of the paper:

> **GraVLM: A Hierarchical Graph-Aligned Vision-Language Model for Cross-Modal Retrieval in Remote Sensing**  
> *Anonymous*  
> Submitted to *Anonymous*, 2026.

---

## 🔍 Overview

**GraVLM** the first hierarchical graph-aligned vision-language framework for RS cross-modal retrieval.
---

## 📁 Project Structure

```
GraVLM/
├── det_json/                          # Detected object annotations (rotated bounding boxes)
├── models/                            # Pre-trained CLIP model weights
├── test/                              # Evaluation scripts and sample test data
├── triples/                           # Extracted entity-relation triplets for ERG construction
├── clip.py                            # CLIP feature extraction wrapper
├── clip_gcn_image_feature_integration.py  # Integrate CLIP features with visual graph via GCN
├── dataloader.py                      # Data loading pipeline (image-text pairs)
├── evaluate.py                        # Performance evaluation (Recall@K, etc.)
├── generate_adjacency_degree_matrices.py.ipynb  # Graph adjacency preprocessing demo
├── infer.py                           # Inference script for retrieval testing
├── loss.py                            # Loss functions (HGAC, contrastive, etc.)
├── model.py                           # Core GraVLM model definition
├── simple_tokenizer.py                # Lightweight tokenizer for textual inputs
├── train.py                           # End-to-end training script
├── train-t.py                         # Text graph (ERG) training script
├── train-v.py                         # Visual graph (SOG) training script
├── T_Graph.py                         # Textual graph construction module
├── V_Graph.py                         # Visual graph construction module
├── T-Graph.ipynb                      # ERG construction visualization notebook
├── V-Graph.ipynb                      # SOG construction visualization notebook
└── requirements.txt                   # Python dependencies
```

---

## 🛠 Environment Setup

- **Python**: 3.8
- **PyTorch**: ≥ 1.10
- Other dependencies are listed in `requirements.txt`

### Install dependencies

```bash
# (Optional) Create and activate a virtual environment
python -m venv fgcross-env
source fgcross-env/bin/activate        # For Linux/macOS
# fgcross-env\Scripts\activate         # For Windows

# Install required packages
pip install -r requirements.txt
```

---

## 📦 Datasets

The following publicly available remote sensing datasets are supported:

| Dataset    | Description                                               | Link                                                                                                          |
| ---------- |-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **RS5M**   | Large-scale pretraining dataset with 5M image-text pairs  | https://github.com/om-ai-lab/RS5M                                                                             |
| **RSITMD** | Remote Sensing Image-Text Multimodal Dataset              | [https://github.com/ucas-vg/RSITMD](https://github.com/ucas-vg/RSITMD)                                        |
| **RSICD**  | Remote Sensing Image Captioning Dataset                   | [https://github.com/ucas-vg/RSICD-official](https://github.com/ucas-vg/RSICD-official)                        |
| **UCM**    | UC Merced Land Use Dataset                                | [http://weegee.vision.ucmerced.edu/datasets/landuse.html](http://weegee.vision.ucmerced.edu/datasets/landuse.html) |


> 📌 Please download and extract the datasets. Then modify the paths in `dataloader.py` to point to the correct locations.

---

## 📥 Pretrained Models

Download pretrained model weights from Baidu NetDisk:

- 🔗 [https://pan.baidu.com/s/1_557f33eRK_rV5N5qHlwjA?pwd=cxcx]
- 📎 Access Code: `cxcx`

Extract the model files into the `models/` directory.

---

## 🚀 How to Use

### 1. Train the Model

```bash
python train.py
```

### 2. Run Inference

```bash
python infer.py

---> example
Similarity Score Matrix (Text-Image):
              Img1   Img2   Img3   Img4   Img5
Query 1:     0.89   0.21   0.35   0.15   0.09
Query 2:     0.22   0.94   0.13   0.33   0.11
Query 3:     0.14   0.09   0.87   0.25   0.20
Query 4:     0.18   0.22   0.31   0.91   0.28
Query 5:     0.05   0.11   0.18   0.26   0.90
```

### 3. Evaluate Retrieval Performance

```bash
python evaluate.py
```
---

## 📜 License

This project is licensed under the MIT License.

```
MIT License
Anonymous

```

---

## 📚 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{
  title={GraVLM: A Hierarchical Graph-Aligned Vision-Language Model for Cross-Modal Retrieval in Remote Sensing},
  author={Anonymous},
}
```

---

## 📬 Contact

Anonymous