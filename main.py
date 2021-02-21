import argparse
import numpy as np
import torch
import logging
import os

from utils import setup_output_dir, read_from_yaml, disp_params
from model import RMP
from data import Language, Batcher

parser = argparse.ArgumentParser(description="Bilingual Lexicon Induction")
parser.add_argument('-cf', '--config_file',
                    action="store", dest="config_file",
                    type=str, help="path to the config file",
                    required=True)
parser.add_argument('-cuda', '--cuda',
                    action="store", dest="cuda", type=int,
                    default=0, help="<0 for cpu, >= 0 for gpus",
                    required=False)
parser.add_argument('-l', '--log', action="store",
                    dest="loglevel", type=str, default="DEBUG",
                    help="Logging Level")
parser.add_argument('-s', '--seed', action="store",
                    dest="seed", type=int, default=-1,
                    help="use fixed random seed")

def load_batcher(data_params, cuda):
    languages, Lang_name = [], []
    # Load the data into languages
    data_dir = data_params['data_dir']
    for w in data_params['languages']:
        lang = Language(
            name=w['name'],
            cuda=cuda,
            mode=data_params['mode'],
            mean_center=data_params['mean_center'],
            unit_norm=data_params['unit_norm']
        )
        Lang_name.append(w['name'])
        lang.load(w['filename'], data_dir, max_freq=data_params['max_freq'])
        languages.append(lang)
    batcher = Batcher(languages)
    if 'supervised' in data_params:
        filename = data_params['supervised']['fname']
        random = data_params['supervised']['random']
        max_count = data_params['supervised']['max_count']
        if data_params["data_dir"] == "./muse_data/":
            sup_dir_name = os.path.join(data_dir, "crosslingual", "dictionaries")
        elif data_params["data_dir"] == "./vecmap_data/":
            sup_dir_name = os.path.join(data_dir, "dictionaries")
        batcher.load_from_supervised(
            filename, Lang_name[0], Lang_name[1],
            sup_dir_name, random = random, max_count=max_count)
    return batcher

def adaptLanguage(data):
    if data["data_params"]["data_dir"] == "./muse_data/":
        data["data_params"]["languages"][0]["filename"] = "wiki.{}.vec".format(data["src"])
        data["data_params"]["languages"][0]["name"] = data["src"]
        data["data_params"]["languages"][1]["filename"] = "wiki.{}.vec".format(data["tgt"])
        data["data_params"]["languages"][1]["name"] = data["tgt"]
        data["data_params"]["supervised"]["fname"] = "{}-{}.0-5000.txt".format(data["src"], data["tgt"])
        return data
    elif data["data_params"]["data_dir"] == "./vecmap_data/":
        data["data_params"]["languages"][0]["filename"] = "embeddings/{}.emb.txt".format(data["src"])
        data["data_params"]["languages"][0]["name"] = data["src"]
        data["data_params"]["languages"][1]["filename"] = "embeddings/{}.emb.txt".format(data["tgt"])
        data["data_params"]["languages"][1]["name"] = data["tgt"]
        data["data_params"]["supervised"]["fname"] = "{}-{}.train.txt".format(data["src"], data["tgt"])
        return data

params = parser.parse_args()
config = read_from_yaml(params.config_file)
config = adaptLanguage(config)
if params.seed > 0:
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.gpu:
        torch.cuda.manual_seed(params.seed)
devive = "cpu" if params.cuda < 0 else "cuda:{}".format(params.cuda) 
logger = logging.getLogger()
                
model, data_params, model_params = config['method'], config['data_params'], config['model_params']
data_params['output_dir'] = os.path.join(data_params['output_dir'], model + "/" + config['src'] + "-" + config['tgt'])
output_dir, config = setup_output_dir(data_params['output_dir'], config, params.loglevel)
disp_params(data_params, "data_params")
disp_params(model_params, "model_params")
batcher = load_batcher(data_params, params.cuda)

if model == 'RMP':
    MODEL_unDSSBLI = RMP(config['src'], config['tgt'], params.cuda, params.seed, batcher, data_params['data_dir'], output_dir)
    MODEL_unDSSBLI.train(**model_params)  


