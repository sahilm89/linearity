# This is the toplevel analysis file fo the linearity project.
#!/usr/bin/bash

SCRIPT=$(readlink -f $BASH_SOURCE)
SCRIPT_PATH=$(dirname $SCRIPT)

DATA_PATH=/media/sahil/NCBS_Shares_BGStim/patch_data/voltage_clamp_files.txt
threshold=-0.045
while read DATA
    do
        #python DN_sahil_currentFits_noGUI.py $DATA spikes $threshold
        python DN_sahil_currentFits_noGUI.py $DATA nospikes
    done <$DATA_PATH 


#    paste -d'\n' $DATA_PATH $DATA_PATH | while read DATA && read threshold; do
#    python -c "print("DN_sahil_currentFits_noGUI.py $DATA spikes $threshold")"
#    done
