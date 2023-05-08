# MMLAL
Build on top of "Semantic Segmentation with Active Semi-Supervised Representation Learning" [1] with code provided by author Aneesh Rangnekar.

![Model Diagram](/images/diagram.png)

#### Files:
1. main_al.py - File to run active learning approach of S4AL+.
2. main_ssl.py - File to run self supervised learning approach of S4AL+.
3. core_models.py - File that contains all S4AL+ models.
4. get_stats.py - Model to test a specific S4AL+ model.
5. CLIPtest.py - Model to test the CLIPSeg model.
6. ensemble.py - Model that tests the combined model of S4AL+ and CLIPSeg.
7. helpers/ - Folder that contains files relating to dataset and dataloader and model.
8. networks/ - Folder that contains all backbone model codes. Currently ResNet101 and MobileNetV2 are supported. MobileViT support is in the works.
9. datasets/ - Folder that contains the datasets used for training and testing.

#### Setup:
1. Download and extract the dataset from [here](https://drive.google.com/file/d/1_UpN0msa-D999lO40l0LPOKC5M877dU2/view?usp=sharing) and place it in datasets/
2. Download and extract the model weights from [here](https://drive.google.com/file/d/1wkFE18JyKxLRrDipuCs9tt6WEbgvC9tk/view?usp=sharing) and place it in lightning_logs/
3. Download and extract backbone weights from [here](https://drive.google.com/file/d/1P_x1r2afsr7L0bkwBOrTiNpl9JXdpDAw/view?usp=sharing) and place it in networks/
4. Environment Setup:
    1. <code>conda create -n mmlal python=3.10</code>
    2. <code>conda activate mmlal</code>
    3. <code>bash requirements.sh</code>

<b>Note:</b> Your system must have a CUDA enabled GPU for the environment setup.

#### Testing Instructions:

For CLIP: <code>python CLIPtest.py</code>

For S4AL+: <code>python get_stats.py</code>

For MMLAL: <code>python ensemble.py</code>

#### References:

[1] Rangnekar, Aneesh, Christopher Kanan, and Matthew Hoffman. "Semantic Segmentation with Active Semi-Supervised Representation Learning." arXiv preprint arXiv:2210.08403 (2022).
