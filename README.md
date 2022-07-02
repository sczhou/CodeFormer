<p align="center">
  <img src="assets/CodeFormer_logo.png" height=110>
</p>

## Towards Robust Blind Face Restoration with Codebook Lookup Transformer

[Paper](https://arxiv.org/abs/xxx) | [Project Page](https://shangchenzhou.com/projects/CodeFormer/) | [Video](https://youtu.be/d3VDpkXlueI)

<!-- <a href="https://colab.research.google.com/drive/1m52PNveE4PBhYrecj34cnpEeiHcC5LTb?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> -->

[Shangchen Zhou](https://shangchenzhou.com/), [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Chongyi Li](https://li-chongyi.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/) 

S-Lab, Nanyang Technological University

<img src="assets/network.jpg" width="800px"/>

### Updates

<!-- Test code and Colab demo for Face Restoration is available now. -->
- **2022.07.02**:  :star:We will release the test code and model by July 15th. Thanks for your attention! :blush:
- **2022.06.21**:  This repo is created.



#### Face Restoration

<img src="assets/restoration_result1.png" width="400px"/> <img src="assets/restoration_result2.png" width="400px"/>
<img src="assets/restoration_result3.png" width="400px"/> <img src="assets/restoration_result4.png" width="400px"/>

#### Face Color Enhancement and Restoration

<img src="assets/color_enhancement_result1.png" width="400px"/> <img src="assets/color_enhancement_result2.png" width="400px"/>

#### Face Inpainting

<img src="assets/inpainting_result1.png" width="400px"/> <img src="assets/inpainting_result2.png" width="400px"/>



<!-- ### Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer

# create new anaconda env
conda create -n codeformer python=3.8 -y
source activate codeformer

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib
```

### Quick Inference

##### Download Pre-trained Models:
Download the dlib pretrained models from [[Google Drive]([xx](https://drive.google.com/drive/folders/1f24f1UqNY8OyeE9aWxswCKQbR7fe8yz4?usp=sharing)) | [OneDrive](xx)] to the `weights/dlib` folder. 
You can download by run the following command OR manually download the pretrained models.
```
python scripts/download_pretrained_models.py dlib
```

Download the CodeFormer pretrained models from [[Google Drive]([xx](https://drive.google.com/drive/folders/1f24f1UqNY8OyeE9aWxswCKQbR7fe8yz4?usp=sharing)) | [OneDrive](xx)] to the `weights/CodeFormer` folder. 
You can download by run the following command OR manually download the pretrained models.
```
python scripts/download_pretrained_models.py CodeFormer
```

##### Prepare Testing Data:
You can put the testing images in the `inputs/TestWhole` folder. If you would like to test on cropped and aligned faces, you can put them in the `inputs/cropped_faces` folder.


##### Testing for Face Restoration:
```
python inference_codeformer.py --w 0.5 --test_path [input folder]
```

NOTE that we set *w* to  -->

### Citation
If our work is useful for your research, please consider citing:

    @article{zhou2022codeformer,
        author = {Zhou, Shangchen and Chan, Kelvin C.K. and Li, Chongyi and Loy, Chen Change},
        title = {Towards Robust Blind Face Restoration with Codebook Lookup TransFormer},
        journal = {arXiv preprint arXiv:2206.11253},
        year = {2022}
    }

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

### Acknowledgement

This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR).
