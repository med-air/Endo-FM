## Usage

**Note**: Currently, for video-based methods, one GPU could only hold 1 image. Do not put 2 or more images on 1 GPU!



### Data preparation

Please download [CVC-Clinic](https://giana.grand-challenge.org/polypdetection/) and [ASUMayo](https://polyp.grand-challenge.org/AsuMayo/) endoscopic video datasets. After that, we recommend converting the mask images to PASCAL VOC XML annotations (scripts refer to [mask_to_voc.py](datasets/mask_to_voc.py)) and symlinking the converted data path to `datasets/`. The path structure should be as follows:

  ```none
  STFT
  ├── datasets
  │   ├── ASUVideo
  │   │   ├── Annotations
  │   │   │   ├── subdir-annos
  │   │   ├── Data
  │   │   │   ├── subdir-images
  │   │   ├── ImageSets
  │   │   │   ├── ASUVideo_train_videos.txt
  │   │   │   ├── ASUVideo_val_videos.txt
  │   ├── CVCVideo
  │   │   ├── Annotations
  │   │   │   ├── subdir-annos
  │   │   ├── Data
  │   │   │   ├── subdir-images
  │   │   ├── ImageSets
  │   │   │   ├── CVCVideo_train_videos.txt
  │   │   │   ├── CVCVideo_val_videos.txt

  ```

**Note**: Since the two databases are protected by copyright, we only uploaded the preprocessed data of one subdirectory in the test dataset at [drive](https://drive.google.com/file/d/1vDshjsgPV8FbO-gpX9LC2elhtClv5Nnr/view?usp=sharing) for data references and model inference. In order to quickly test our model, we have already provided the image list of this subdirectory as a txt file under directory `datasets/ASUVideo/ImageSets`. If you are interested in using these databases, please contact the copyright owner.



### Inference

The inference command line for testing on the ASUMayo validation set with 1 GPU:

    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        tools/test_net.py \
        --master_port=$((RANDOM + 10000)) \
        --config-file configs/STFT/asuvid_R_50_STFT.yaml \
        MODEL.WEIGHT pretrained_models/ASUMayo_STFT_R_50.pth \
        OUTPUT_DIR log_dir/asuvid_R_50_STFT \
        TEST.IMS_PER_BATCH 1
    
Please note that:
1) 1 GPU only holds 1 images for STFT, you should keep TEST.IMS_PER_BATCH equal to the number of GPUs you use.
2) The pretrained model of STFT on the ASUMayo dataset is available at [here](https://drive.google.com/file/d/1CCpf0EF5FgYz1mTo1YfAyqoRR0s0PM15/view?usp=sharing). After downloaded, it should be placed at `pretrained_models/`.
3) If you want to record the detailed results, please specify `OUTPUT_DIR`. Meanwhile, you can visualize the test results by adding `--visulize` option.
4) If you want to evaluate a different model, please change `--config-file` and `MODEL.WEIGHT`.



### Training

The following command line will train asuvid_R_50_STFT on 4 GPUs:

    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        tools/train_net.py \
        --master_port=$((RANDOM + 10000)) \
        --config-file configs/STFT/asuvid_R_50_STFT.yaml \
        OUTPUT_DIR log_dir/asuvid_R_50_STFT
        
Please note that:
1) The models will be saved into `OUTPUT_DIR`.
2) If you want to train other methods, please change `--config-file`.
3) For training FGFA, the pretrained weight of FlowNet is available at [here](https://drive.google.com/file/d/1gib7XtS1fSYDTM9RnUJ72a3vREV_6SJH/view). After downloaded, it should be placed at `pretrained_models/`.



### Customize
If you want to use these methods on your own dataset or implement your new method. Please refer to [here](https://github.com/Scalsol/mega.pytorch/blob/master/CUSTOMIZE.md).
