find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type f -name '*pkl'|while read cell ; do 
cp $cell /home/sahil/Documents/Codes/linearity/data/
done
