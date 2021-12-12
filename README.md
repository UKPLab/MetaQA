[![arXiv](https://img.shields.io/badge/arXiv-2112.01922-b31b1b.svg)](https://arxiv.org/abs/2112.01922)
# MetaQA: Combining Expert Agents for Multi-Skill Question Answering

In this work, we propose to combine expert agents with a novel, flexible, and training-efficient architecture that considers questions, answer predictions, and answer-prediction confidence scores to select the best answer among a list of answer candidates. Through quantitative and qualitative experiments we show that our model i) creates a collaboration between agents that outperforms previous multi-agent and multi-dataset approaches in both in-domain and out-of-domain scenarios, ii) is extremely data-efficient to train, and iii) can be adapted to any QA format.

## Setup (data, pretrained model, QA Agent's Predictions)
Create a conda environment for Python 3.8 and install PyTorch 1.8.2 (LTS)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

Install the requirements with:

```
pip install -r requirements.txt
```

We have made a script `setup.sh` that downloads all the necessary data to run this repository. This script creates three foldres: `data`, `MetaQA_models`, and `qa_agents`, downloads and unzips the following files:

- Model checkpoint: https://public.ukp.informatik.tu-darmstadt.de/metaqa/MetaQA.zip
- Agent predictions: https://public.ukp.informatik.tu-darmstadt.de/metaqa/qa_agents.zip
- QA Datasets: https://public.ukp.informatik.tu-darmstadt.de/metaqa/data.zip

```
sh setup.sh
```

## Configuration
We include a `config.yaml` file to configure the datasets and agents you want to use.
Include the datasets in they key `datasets`. If you want to add new datasets, you need  to add a dataset class in MetaQA/QA_Dataset/QA_Dataset.py to preprocess the dataset. 

You also need to configure the data paths in config.yaml
- agents_path: the folder where the predictions of the agents are store. You should download here 
- output_path: output path to store the metaqa model you train
- train_data_path: path to the training sets (questions, context, answers)
- dev_data_path: path to the dev sets (questions, context, answers)
- test_data_path: path to the test sets (questions, context, answers)

## How to run

## Training

```
python MetaQA.py --do_train --do_test \
                 --model_name my_metaqa \
                 --training_sample_size 10000 \
                 --config config.yaml \
```

## Inference

```
python MetaQA.py --do_test \
                 --pretrained_metaqa_path path_to_pretrained_model \
                 --config config.yaml \
```


## Citation
If you find our work relevant, please use the following citation:
```
@article{Puerto2021MetaQACE,
  title={MetaQA: Combining Expert Agents for Multi-Skill Question Answering},
  author={Haritz Puerto and Gozde Gul cSahin and Iryna Gurevych},
  journal={ArXiv},
  year={2021},
  volume={abs/2112.01922}
}
```

## Contact

Haritz Puerto

https://haritzpuerto.github.io

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.