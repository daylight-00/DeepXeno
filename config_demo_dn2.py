import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file

config = {
    "chkp_name"         : "demo_deepneo",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",
    "seed"              : 100,
    
    "model"             : model.DeepNeo_2_Custom,
    "model_args"        : {
    },
    "encoder"           : encoder.deepneo_2,
    "encoder_args"      : {
    },
    "CrossValidation": {
        "num_folds"     : 5,
    },
    "Data": {
        "epi_path"      : "~/project/IMG/data/final/mhc2_full_human_train.csv",
        "epi_args"      : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "hla_path"      : "~/project/IMG/data/final/HLA2_IMGT_light.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "seperator" : ",",
        },
        "test_path"     : "~/project/IMG/data/final/tcell_human_train_set.csv",
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
        "num_epochs"    : 100,
        "patience"      : 10,
        "regularize"    : True,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCELoss,
        "optimizer"     : optim.AdamW,
        "optimizer_args": {
            "lr"        : 1e-5,
            # "weight_decay"  : 0.01,
        },
        "use_scheduler" : False,
        "chkp_prefix"   : "best",
        "transfer"      : True,
    },
    
    "Test": {
        "batch_size"    : 1024,
        "chkp_prefix"   : "best",
        "feat_path"     : "feat.h5",
        "target_layer"  : "fc",
    },
}