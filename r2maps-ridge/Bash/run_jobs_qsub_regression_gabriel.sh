# How to run this script: 
# 
#    cd 
#    ./run_jobs.sh  <GROUP>

for SUBJECT in {1..3}
do
	filename_bash=RunScripts/log_$SUBJECT.sh
	filename_py='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/Code/alignment_model/main.py'$SUBJECT
	output_log='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Logs_regression/output_log_o'$SUBJECT
	error_log='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LSTM/Logs_resgression/output_log_e'$SUBJECT
	queue='Unicog_long'
	job_name='Subject_'$SUBJECT
	walltime='72:00:00'

	rm -f $filename_bash
	touch $filename_bash
	echo "python2.7 $filename_py" >> $filename_bash
		 
	qsub -q $queue -N $job_name -l walltime=$walltime -o $output_log -e $error_log $filename_bash
		 
done

