# This is the toplevel fitting file for current data.
#!/usr/bin/bash

SCRIPT=$(readlink -f $BASH_SOURCE)
SCRIPT_PATH=$(dirname $SCRIPT)

DATA_PATH=/media/sahil/NCBS_Shares_BGStim/patch_data/voltage_clamp_files_pickles.txt

while read DATA
    do
            python fitting_currents.py $DATA
    done <$DATA_PATH
