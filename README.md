# MMLAL
Build on top of "Semantic Segmentation with Active Semi-Supervised Representation Learning" [1] with code provided by author Aneesh Rangnekar.

Files:
1. main_al.py - File to run active learning approach of S4AL+
2. main_ssl.py - File to run self supervised learning approach of S4AL+
3. core_models.py - File that contains all S4AL+ models
4. get_stats.py - Model to test the latest S4AL+ model
5. CLIPtest.py - Model to test the CLIPSeg model
6. ensemble.py - Model that tests the combined model of S4AL+ and CLIPSeg
7. helpers/ - Folder that contains files relating to dataset and dataloader and 
References:

[1] Rangnekar, Aneesh, Christopher Kanan, and Matthew Hoffman. "Semantic Segmentation with Active Semi-Supervised Representation Learning." arXiv preprint arXiv:2210.08403 (2022).
