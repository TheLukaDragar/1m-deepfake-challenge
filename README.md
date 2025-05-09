
## Setup the Dataset

To ensure the Docker container has access to the full 1MDF dataset, you need to mount the dataset directory to the container. Follow these steps:

1. Prepare the Dataset: Make sure you have the full 1MDF dataset available on your local machine.

2.	Mount the Dataset Directory:

    Modify the ```docker-compose.yml``` file to mount the dataset directory to ``/1mdfFinal/dataset/`` inside the container. 

#### Checkpoints
The visual model the checkpoint is big you can download it here 
https://drive.google.com/file/d/116agYsOSV-XJYXGCm1PEat8CW4zzu1ui/view?usp=sharing

after downloading mount it in the docker container to ```/1mdfFinal/visual/models_luka_1mdf/eva_giant_patch14_224.clip_ft_in1k/jn41pnlb/eva_giant_patch14_224.clip_ft_in1k_final.ckpt``` by modifying the ```docker-compose.yml``` file

The audio checkpoint is smaller and is already in the image.

## Docker setup

As requested the code is meant to be run in a docker continer. To build the docker image run the following command:

```bash
docker compose -f "docker-compose.yml" up -d --build 
```

This will build the docker image and start the container. The image is based on the official PyTorch image it also creates a conda enviroment with all the required packages to run the code. The initial setup will take some time to download the required packages.


## Before running 

1.Start the Docker Container, run the following command to start the Docker container:

```bash
docker-compose up
```

2. Access the Container, run the following command to access the container:

```bash
docker-compose exec pytorch_container bash
```

3. Activate the conda enviroment, once you are in the container run the following command to activate the conda enviroment:

```bash
source /opt/conda/bin/activate pytorch_env_1mdf && conda activate pytorch_env_1mdf
```

Now you are ready to run the code.

## Evaluation
#### Audio

Run the following command to run predictions on the first 100 videos in the test set this creates predictions using the ``TDL_res2-epoch=31-val_loss=0.02-train_loss=0.02.ckpt`` model.

```bash
cd audio
python3 predict.py --videos_txt ../dataset_helper/test_100.txt
```


For full evaluation run the following command:

```bash
python3 predict.py --videos_txt ../dataset_helper/test_files.txt
```

The predictions are stored as pkl files in the ``audio/audio_predictions`` folder.

Make sure you delete the ``audio/audio_predictions`` folder before running the evaluation again.


#### Video


Run the following command to run predictions on the first 100 videos in the test set this creates predictions using the ``/visual/models_luka_1mdf/eva_giant_patch14_224.clip_ft_in1k/jn41pnlb/eva_giant_patch14_224.clip_ft_in1k_final.ckpt`` model checkpoint.

```bash
cd visual
python3 predict.py --videos_txt ../dataset_helper/test_100.txt
```

for full evaluation run the following command:

```bash
cd visual
python3 predict.py --videos_txt ../dataset_helper/test_files.txt
```

The frame level predictions are stored as pkl files in the ``visual/visual_predictions`` folder.


### Segment extraction and merging

To transform frame level predictions into video level predictions and visualize the results 
use the notebook ``segment_extraction_and_merging.ipynb``, please modify the paths to the predictions to use our predictions or the ones you have generated and install the required packages with the following command:

```bash
python3 -m pip install jupyter
```

```bash
jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

and open the ``segment_extraction_and_merging.ipynb`` in the browser.




or script ``segment_extraction_and_merging.py``.

```bash
python3 segment_extraction_and_merge.py
```


This script takes the predictions from the audio and video models then uses the segment extraction algorithms to create segments then it merges them and transforms in the submission format.

The end result is saved in as ```1mdfFinal/submissionmerged_roundedtozero2_avg_frmes_predictions_avgpred_raw.json```.


For task 1 one can use a trained visual model (we can provide the checkpoint in advance - due to upload size, code in ```task1_train_timm1m_random_og_adapted_to_task1.py```) and get auc close to 1
or for the best final submited result using ffmpeg
```bash
python3 task1_unintended.py
```

this creates the ```task1_predictions.txt```



## Training 

#### Audio

First we need to extract the Wav2Vec features from the audio files this creates another directory in dataset called ``train_audio``.
    
```bash
cd audio
python3 preprocess.py --pkl_path ../dataset_helper/train_100.pkl
```

now we run the algoritm for window selection:

```bash
python3 train_label_generation_res2.py --pkl_path ../dataset_helper/train_100.pkl --output_pkl ./train_audio_modifed_and_real_segments_res2_100.pkl
```

or the same setup but for the full dataset

```bash
cd audio
python3 preprocess.py --pkl_path ../dataset_helper/all_train.pkl
python3 train_label_generation_res2.py --pkl_path ../dataset_helper/all_train.pkl --output_pkl ./train_audio_modifed_and_real_segments_res2.pkl
```


for training on the first 100 videos in the dataset run the following command
    
```bash
cd audio
python3 train_res2.py --pkl_path ./train_audio_modifed_and_real_segments_res2_100.pkl
```

for training on the full dataset run the following command

```bash
cd audio
python3 train_res2.py --pkl_path ./train_audio_modifed_and_real_segments_res2.pkl
```

when you end the traing a ``/audio/models_luka_1mdf_audio_res_2/TDL_res2/run_id/ is created ``


#### Video

Here the training data preprocessing is already done and the pkl files are in the visual folder.

For training the visual model for task 2 use this command:
```bash
cd video
python3 train_timm1m_random_og_real_segments_beetween_fake.py --pkl_path ./only_visual_with_duplicate_so_its_easier_to_load_w_real_segments_beetween_fake_plus_15_percof_real_splited.pkl
```


For training the visual model for task 1 use this command
```bash
cd video
python3 task1_train_timm1m_random_og_adapted_to_task1.py --pkl_path ../dataset/all_train.pkl
```

When you end the training a ``/visual/models_luka_1mdf/eva_giant_patch14_224.clip_ft_in1k/run_id/`` is created


