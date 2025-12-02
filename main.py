import os
import sys
import subprocess
import pandas as pd
from train import Trainer
from config import  Config
from inference import Inference
from data.augment import process_data, split_data, process_one_file
from data.triplet_dataset import TripletDataset


cfg = Config()
data_cfg = cfg.get("data")

#process the data
if cfg.get("process"):
    process_data(cfg, data_cfg)

#train Autoencoders
if cfg.get("train"):
    m_path = os.path.dirname(cfg.get("data","raw_data_dir"))
    train_csv =  m_path + "_train.csv"
    val_csv = m_path + "_val.csv"

    is_df_exist = not os.path.exists(train_csv) or not os.path.exists(val_csv)
    if cfg.get("split") or is_df_exist:
        train_df, val_df = split_data(
            m_path + ".csv",
            cfg.get("data", "split")
        )
        train_df.to_csv(train_csv)
        val_df.to_csv(val_csv)
    else:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

    model = None
    pretrain_data_cfg = cfg.get("data_pretrain")

    if cfg.get("pre_process"):

        process_data(cfg, pretrain_data_cfg)

    if cfg.get("pretrain"):

        pre_m_path = os.path.dirname(cfg.get("data_pretrain","raw_data_dir"))
        pre_train_csv =  pre_m_path + "_train.csv"
        pre_val_csv = pre_m_path + "_val.csv"

        is_df_exist = not os.path.exists(pre_train_csv) or not os.path.exists(pre_val_csv)
        if cfg.get("split") or is_df_exist:
            pre_train_df, pre_val_df = split_data(
            pre_m_path + ".csv",
            cfg.get("data", "split")
        )
            pre_train_df.to_csv(pre_train_csv)
            pre_val_df.to_csv(pre_val_csv)
        else:
            pre_train_df = pd.read_csv(pre_train_csv)
            pre_val_df = pd.read_csv(pre_val_csv)

        pre_train_df, pre_val_df = split_data(
            pre_m_path + ".csv",
            cfg.get("data", "split")
        )

        pre_train_gen = TripletDataset(pre_train_df, cfg, cfg.get("data_pretrain", "dataset_dir"))
        pre_val_gen = TripletDataset(pre_val_df, cfg, cfg.get("data_pretrain", "dataset_dir"))

        pretrainer = Trainer(cfg, pre_train_gen, pre_val_gen)
        model = pretrainer.fit()
    else:
        model = f'{cfg.get("ae","full_ae_path")}_{cfg.get("window","matching")}_{cfg.get("ae", "N_FEATURES")}_{cfg.get("window", "size")}_{cfg.get("window", "stride")}.pth'

    train_gen = TripletDataset(train_df, cfg)
    val_gen = TripletDataset(val_df, cfg)

    trainer = Trainer(cfg, train_gen, val_gen, model)
    trainer.fit()

if cfg.get("inference"):

    m_d = process_one_file(cfg, "data/recorded_sessions/session_20251201_094111.mp4",)
    s_d = process_one_file(cfg, "data/custom/a.mp4",)
    v_d = process_one_file(cfg, "data/ready_train_utd/a25_s6_t1_v0_f1.npy", )

    inf = Inference(cfg)

    score = inf.dtw_similarity(m_d, s_d, vis=True)

    inf.demonstrate(m_d, s_d, v_d)

if cfg.get("webapp"):

    print("🚀 Launching Web App interface...")
    
    # Assuming you saved the previous code as 'app.py' in the same folder
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # This command is equivalent to typing "streamlit run app.py" in the terminal
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    
    # Optional: Stop the rest of main.py from running after the app closes
    sys.exit()