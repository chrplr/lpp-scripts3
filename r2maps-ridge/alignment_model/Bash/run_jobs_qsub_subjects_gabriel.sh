# How to run this script: 
# 
#    cd 
#    ./run_jobs.sh  <GROUP>

for ROI in {0..5}
do
    filename_bash=RunScripts/log_$ROI.sh
	filename_py='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Code/alignment_model/main.py '$ROI
	output_log='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Logs/output_log_o'$ROI
	error_log='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Logs/output_log_e'$ROI
	queue='Unicog_long'
	job_name='ROI'$ROI
	walltime='72:00:00'

	rm -f $filename_bash
	touch $filename_bash
	echo "python2.7 $filename_py" >> $filename_bash
         
	qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log $filename_bash
         
done

