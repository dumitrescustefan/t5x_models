# Romanian T5x models training repo 

This repo keeps the setup and training scripts to train [T5x](https://github.com/google-research/t5x) models on TPUs.  

The first section of the readme presents our trained models, the second section discusses the training scripts while the third section discusses the steps needed to setup the environment on Google Cloud, the actual training and monitoring and the final checkpoint conversions to PyTorch. 

## Section 1 - Models and results

We have 4 models trained so far, a base and a large version of a T5v1.1 and an mT5.

(results soon)

## Section 2 - T5x training scripts

Note that we only do pretraining, i.e. span corruption, as we did not have any supervised tasks at the moment of training.
Also note that the files here were changed many times on-site, like when a model trained to 2M steps, and we decided to train in for another 1M and then another 1M, so you need to adjust params to your liking. 

In essence, a model requires 4 files for training:

1. ``ro_<model>_<size>.gin`` 
2. ``ro_<model>_<size>_tasks.py``
3. ``ro_<model>_<size>_pretrain.gin``
4. ``ro_<model>_<size>.sh``

Let's take an example and say we want to train a T5x base model from scratch. Thus, we have model=t5x and size=base. 

1. In ``ro_t5x_base.gin``:
   * notice the second line with the reference to the gin file. Change accordingly to match filename. Same for the import of the python file.
   * in this case we train from scratch with our vocab so we set:
```bash
network.T5Config:
  vocab_size = 64000

# Vocabulary (shared by encoder and decoder)
VOCABULARY = @seqio.SentencePieceVocabulary()
seqio.SentencePieceVocabulary.sentencepiece_model_file = "<bucket path to sentencepiece model file>"
```

    * take care that the ``MIXTURE_OR_TASK_NAME`` is set to the correct task found in the tasks.py file
    * ``TASK_FEATURE_LENGTHS = {"inputs": 512, "targets": 256}`` means that the encoder has the input size 512 and the decoder will generate up to 256 tokens.
    * ``TRAIN_STEPS = 2_000_000`` set to the number of steps you want to train to.
    * to start from an existing chekpoint (like we did for mT5), set ``INITIAL_CHECKPOINT_PATH = "gs://t5-data/pretrained_models/t5x/mt5_base/checkpoint_1000000"``
    * set ``PjitPartitioner.num_partitions = 2`` to a higher number if you get out of memory, and decrease the TASK_FEATURE_LENGTHS. (see official t5x repo)

2. In ``ro_t5x_base_tasks.py``:
   * set vocabulary with ``vocabulary = seqio.SentencePieceVocabulary('gs://myv4-bucket/sentencepiece/ro.model', extra_ids=0)``
   * adapt the following script in the final part of the file:
   
```python
dataset_name = 'dumitrescustefan/rlm'
dataset_params = {"path": dataset_name, "use_auth_token": True, "streaming": True}
dataset_shapes = None
TaskRegistry.add(
    "rlm_span_corruption_stream",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_params=dataset_params),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=dataset_shapes,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,        
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": DEFAULT_OUTPUT_FEATURES["targets"]},
    metric_fns=[]
)
```
   * set the dataset name, and use_auth_token to True if the dataset is private 
   * set streaming to True as we don't have to deal with disk space
   * take care of the task name, as it's referenced in the gin file above

3. In ``ro_t5x_base_pretrain.gin``: 
   * set ``BATCH_SIZE`` as large as possible 
   * in ``utils.SaveCheckpointConfig:`` set period to 100K so it saves every 100K steps, and ``keep`` to keep the last number of checkpoints.

4. In ``ro_t5x_base.sh``:
   * ``PROJECT_DIR=${HOME}"/models/t5x_models"`` note where you save the models (see Section 3)
   * ``T5X_DIR="../../t5x"`` directory where the t5x is cloned (see Section 3)
   * ``MODEL_DIR="gs://myv4-bucket/ro_t5x_base"`` where to save the checkpoints and all the training logs and other files (bucket)
   * ``GIN_FILE="ro_t5x_base.gin"`` link to the gin file above


## Section 3 - Hands-on: TPU setup, training, monitoring and checkpoint conversion  

The setup assumes you already have:
1. a corpus in the HuggingFace dataset format already uploaded and available
2. a sentencepiece vocabulary if you want to train from scratch (the mT5x models use the vocab file on the public bucket)

#### TPU and bucket creation 

First, set up a bucket where you'll keep all the training output. Always use a bucket in the same zone as the TPUs. I've done this in GCP's web UI very fast. If you have trouble accessing the bucket from the TPUs, check that the service account has access to the bucket, and add all the Storage Legacy and Storage Owner permissions to the service account of the TPU created in the next step.  

Next, let's create a TPU VM. For a v4-8, run the following in the GCP console:
```bash
gcloud alpha compute tpus tpu-vm create <tpu_name> --zone <zone> --accelerator-type v4-8 --version v2-alpha-tpuv4 --subnetwork=tpusubnet
```
For anything larger than a v4-8: 
```bash
gcloud alpha compute tpus tpu-vm create <tpu_name> --zone <zone> --accelerator-type v4-32 --version v2-alpha-tpuv4-pod --subnetwork=tpusubnet
```
Note that for a TPU v3 use the ``v2-alpha`` image, without the ``--subnetwork`` param. I've only run this on v4s, so can't make guarantees the training will work.

To ssh into a TPU VM:
```bash
gcloud alpha compute tpus tpu-vm ssh <tpu_name> --zone <zone>
```

*!!!!!!!!!!!!!!!!!!!!!!!!!!!!*

What I wish somebody would have told me earlier (thank Per!) and it's not written in GCP or at least does not stand out, is that for *anything larger than a v8*, like a v16 or v32, **the TPU is actually v8s wrapped together**.

This means that a v16 is 2*v8s, and *you have to ssh and run the training script into each v8 slice!!*

For example, a v32 has 4 slices; to ssh into each, use the ``--worker`` param, starting from 0. To ssh into the fourth v8 slice, run:
```bash
gcloud alpha compute tpus tpu-vm ssh <tpu_name> --zone <zone> --worker 3
```

*!!!!!!!!!!!!!!!!!!!!!!!!!!!!*

#### TPU setup

After ssh-ing into *each* pod slice, run directly: 

```bash
# replace with your GCP project and zone where the TPU was created
printf "1\n<project_name>\nY\n<zone>\n" | gcloud init

sudo apt-get update && sudo apt install -y git-lfs mc sshfs htop

git lfs install
pip install -U pip
git clone --branch=main https://github.com/google-research/t5x
sudo pip uninstall jax jaxlib libtpu-nightly libtpu libtpu-tpuv4 -y
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.htm
pip install datasets
python3 -m pip install -e 't5x/.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install tensorflow==2.9.0

# set git email and name 
git config --global user.email "<email>" && git config --global user.name "<name>" && git config --global credential.helper store

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
echo "export PATH='~/.local/bin:$PATH'" >> ~/.bashrc && source ~/.bashrc

# if you use private corpora you need to copy-paste a token here 
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('<hf_token_obtained_from_the_hf_website>')"

mkdir models
cd models

# use your repo here, note the name that needs to match with the training scripts (here is t5x_models)
git clone https://github.com/dumitrescustefan/t5x_models.git
cd t5x_models
```
 

#### Training start

After you've run the setup on the v8 or on each TPU slice (if you run on v16s or larger), then keep a console in each slice and run the starting script:

```bash
git pull && bash ro_t5x_base.sh (your .sh file here)
```

Why this way? Because for example, on a v4-32 there are 4 slices where you need to run the bash file. And almost certainly you'll have some error in the training files. As each slice is separate, you have to copy everything 4 times. So why not better edit it in a single place, git push it, and then on each slice just git pull and run the training script? This is Per's suggestion and it saved us a lot of time!

#### Training monitoring

My way of bare-bone monitoring (as the TPUs v4 were MUCH more stable than the TPUv3s I've played with), is to run this in a Colab:

```python
from google.colab import auth
auth.authenticate_user()
```

This will authenticate you in Colab. Next, let's copy some log files from the bucket and start a tensorboard UI:
```bash
!rm -rf logs/ && mkdir logs
!gsutil -m cp gs://myv4-bucket/ro_t5x_base/train/* logs/

%reload_ext tensorboard
%tensorboard --logdir logs
```

Run this from time to time to see how the training loss evolves. 

On a v4-32, the t5x-large trained to 4M steps in about 3 weeks. On a v4-8, a t5x-base trains 1M steps in less than a week. 

#### Checkpoint conversion

When converting the checkpoints you have to install the following dependency:

``pip3 install --upgrade tensorstore==0.1.13``

In ``convert.sh`` script you will have to modify the path to the folder with checkpoints:

``folder=CHECKPOINTS_PATH``

For example ``folder=mt5x-base``, and after run:

``bash convert.sh``

This will convert checkpoints from Tensorflow to Flax and Pytorch.

## Acknowledgements

Many thanks to the **Tensorflow Research Credits (TRC) team** without which these models would not have been possible to train and opensource. Their support team was quick and helpful throughout the months I've had TRC credits. If only the TPU on-line documentation was as good as their support :)

I've found that the v4s are much more stable than the v3s; maybe it was a capacity issue with the v3s, but considering they were all on-demand, I've never had a v3 run for more than a week or two without some reset and a new IP address. The v4s never crashed even once. Plus, they are a beast of a device: training a 1.2B model for 4M steps in 2-3 weeks on a v4-32 is amazing.

_Yours truly,_ 

_[Stefan Dumitrescu](https://github.com/dumitrescustefan), [Mihai Ilie](https://github.com/iliemihai) and [Per Egil Kummervold](https://huggingface.co/north)_

