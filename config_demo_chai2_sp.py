import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file

config = {
    "chkp_name"         : "demo_chai2_sp",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",
    "seed"              : 100,

    "model"             : model.cat2_alpha_sp,
    "model_args"        : {
        "hla_dim_s"       : 384,
        "hla_dim_p"       : 269,
        "epi_dim_s"       : 384,
        "epi_dim_p"       : 15,
        "hla_nhead_s"     : 8,
        "hla_nhead_p"     : 1,
        "epi_nhead_s"     : 8,
        "epi_nhead_p"     : 5,

    },

    "encoder"           : encoder.plm_plm_sp2,
    "encoder_args"      : {
        "hla_emb_path_s" : "/home/alpha/project/EMB/key_replaced/emb_hla2_chai_single_light_re_1212.h5",
        "epi_emb_path_s" : "/home/alpha/project/EMB/emb_epi_chai_single_light_1212.h5",
        "hla_emb_path_p" : "/home/alpha/project/EMB/key_replaced/emb_hla2_chai_pair_light_re_1212.h5",
        "epi_emb_path_p" : "/home/alpha/project/EMB/emb_epi_chai_pair_light_1212.h5",
    },

    "Data": {
        "epi_path"      : "/home/alpha/project/IMG/data/final/mhc2_full_human_train.csv",
        "epi_args"      : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "hla_path"      : "/home/alpha/project/IMG/data/final/HLA2_IMGT_light.csv",
        
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "seperator" : ",",
        },
        "test_path"     : "/home/alpha/project/IMG/data/final/mhc2_full_human_test.csv",
        "test_args"     : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "num_workers"   : 8,
        "val_size"      : 0.2,
    },

    "Train": {
        "batch_size"    : 128,
        "num_epochs"    : 50,
        "patience"      : 100,
        "regularize"    : False,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCEWithLogitsLoss,
        "optimizer"     : optim.AdamW,
        "optimizer_args": {
            "lr"        : 1e-5,
        },
        "use_scheduler" : False,
    },
    
    "Test": {
        "batch_size"    : 64,
        "chkp_prefix"   : "best",
    },
}