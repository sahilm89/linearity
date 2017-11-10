find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c*'|while read cell ; do 
if ls $cell/plots/*.pkl 1> /dev/null 2>&1; then
   echo "$cell: files do exist"
   cp --parents $cell/plots/*.pkl /home/bhalla/Documents/Codes/data/
else
   echo "$cell: files do not exist"
fi
done
