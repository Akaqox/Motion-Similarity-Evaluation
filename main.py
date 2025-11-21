import os
import pandas as pd
from train import Trainer
from config import  Config
from inference import Inference
from data.augment import process_data, split_data
from data.triplet_dataset import TripletDataset


cfg = Config()
data_cfg = cfg.get("data")

#process the data
if cfg.get("process"):
    process_data(data_cfg)
    
#train Autoencoders
if cfg.get("train"):
    train_csv = cfg.get("data", "train_csv")
    val_csv = cfg.get("data", "val_csv")

    is_df_exist = not os.path.exists(train_csv) or not os.path.exists(val_csv)
    if cfg.get("split") and is_df_exist:
        train_df, val_df = split_data(
            cfg.get("data", "metadata"), 
            cfg.get("data", "split")
        )
        train_csv.to_csv(train_csv)
        val_csv.to_csv(val_csv)
    else:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

    train_gen = TripletDataset(train_df, cfg)
    val_gen = TripletDataset(val_df, cfg)

    trainer = Trainer(cfg, train_gen, val_gen)
    trainer.fit()

if cfg.get("inference"):

    m_d = "data/ready_train/a17_s5_t4_v2.npy"
    s_d = "data/ready_train/a17_s1_t4_v4.npy"
    v_d = "data/ready_train/a25_s6_t1_v0.npy"
    
    inf = Inference(cfg)

    score = inf.dtw_similarity(m_d, s_d)
    print(score[0])
    inf.demonstrate(m_d, s_d, v_d)

   