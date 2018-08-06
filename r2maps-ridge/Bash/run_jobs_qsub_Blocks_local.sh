# How to run this script: 
# 
#    cd 
#    ./run_jobs.sh  <GROUP>
set st = 1
set ed = 9

for BLOCK in {1..9}
do
    filename_bash=RunScripts/log_$BLOCK.sh
	filename_py='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Onset_Amplitude/en/Make_Regressors.py '$BLOCK
	output_log='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Logs/output_log_o'$BLOCK
	error_log='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Logs/output_log_e'$BLOCK
	queue='Unicog_long'
	job_name='Block_'$BLOCK
	walltime='72:00:00'

	#rm -f $filename_bash
	#touch $filename_bash
	#echo "python2.7 $filename_py" >> $filename_bash
         
	python2.7 $filename_py >> $output_log &
	#qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log $filename_bash
         
done

