# This is the toplevel analysis file fo the linearity project.
#!/usr/bin/bash

SCRIPT=$(readlink -f $BASH_SOURCE)
SCRIPT_PATH=$(dirname $SCRIPT)

DATA_PATH=/media/sahil/NCBS_Shares_BGStim/patch_data/voltage_clamp_files.txt

while read DATA
    do
        python analyzeCurrentData.py $DATA
    done <$DATA_PATH
