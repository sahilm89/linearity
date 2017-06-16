# Removing already existing files for updating
rm /media/sahil/NCBS_Shares_BGStim/patch_data/current_clamp_files.txt
rm /media/sahil/NCBS_Shares_BGStim/patch_data/current_clamp_files_with_GABAzine.txt
rm /media/sahil/NCBS_Shares_BGStim/patch_data/voltage_clamp_files.txt
rm /media/sahil/NCBS_Shares_BGStim/patch_data/CA3_files.txt
rm /media/sahil/NCBS_Shares_BGStim/patch_data/IN_voltage_clamp_files.txt

## Checking for CPP under current clamp cells
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c?'|while read cell ; do 
if [ -d "$cell/CPP/" ]; then
   echo $cell
  echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/current_clamp_files.txt
fi
done

## Checking for CPP under cut slices current clamp cells
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c?_CS'|while read cell ; do 
if [ -d "$cell/CPP/" ]; then
   echo $cell
  echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/current_clamp_files.txt
fi
done

## Checking for CPP under current clamp cells with GABAzine
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c?'|while read cell ; do 
if [ -d "$cell/GABAzine/CPP/" ]; then
   echo $cell
  echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/current_clamp_files_with_GABAzine.txt
fi
done

## Checking for CPP under current clamp cells
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name '*_c?_*'|while read cell ; do 
if [ -d "$cell/CPP/" ]; then
   echo $cell
  echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/current_clamp_files.txt
fi
done

## Checking for Interneurons CPP under voltage clamp cells
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c?_IN_EI*'|while read cell ; do 
if [ -d "$cell/CPP/" ]; then
   echo $cell
  echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/IN_voltage_clamp_files.txt
fi
done


## Checking for CPP with spikes under current clamp cells
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c?_CS'|while read cell ; do 
if [ -d "$cell/CPP/" ]; then
   echo $cell
  echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/current_clamp_spikes_files.txt
fi
done

## Checking for CPP under voltage clamp cells
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c?_EI'|while read cell ; do 
if [ -d "$cell/CPP/" ]; then
  echo $cell
  echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/voltage_clamp_files.txt
fi
done


## Checking for CA3 cells
find /media/sahil/NCBS_Shares_BGStim/patch_data/ -mindepth 1 -type d -name 'c?_CA3_CPP'|while read cell ; do 
echo $cell
echo "$cell/">>/media/sahil/NCBS_Shares_BGStim/patch_data/CA3_files.txt
done
