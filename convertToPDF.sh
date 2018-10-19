for file in `/media/sahil/NCBS_Shares_BGStim/Paper/Submissions/Nat_Comm_submission/NComm/Revision/Figures/*.svg`
do
    filename=$(basename -- "$file")
    inkscape $filename.svg --export-pdf= /home/bhalla/Documents/Codes/linearity/Paper_Figures/$filename.pdf
done


