import fileinput

#Dividing punctutaion from other characters

nb_blocks = 1

for i in range(1, nb_blocks + 1):
	block = i
	if block < 10:
		input_data = "../Data/en/block0{}.txt".format(block)
	else:
		input_data = "Block{}.alt.".format(block)
	for i in range(4):
		with fileinput.FileInput(input_data, inplace=True, backup='.bak') as file:
		    for line in file:
		        print(line.replace("  ", " "), end='')

# Add features to modify by copy-pasting fileinput... loops