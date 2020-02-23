python3 ../src/variational_inference.py \
		--inputfile_reviews  '../input/labeled_data.csv'\
		--answer_matrix '../input/answer_matrix.csv'\
		--sup_rate 6\
		--iterr 10\
		--classifier 'svm'\
		--evaluation_file '../output/small_example.csv'\
