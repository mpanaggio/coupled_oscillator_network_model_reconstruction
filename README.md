# learn_network_structure
Code to identify the coupling network from time-series for coupled oscillators


In this project, we simulate a network of coupled oscillators  governed by the Kuramoto model. Then we use simple convolutional neural network to learn the parameters of the model from the simulated data.

This folder contains the following python files:
1. learn_kuramoto_files.py: auxiliary functions for generating data, analyzing results (you should not need to modify this)
2. run_kuramoto_learn.py: script with loop to generate results and save them to file
3. plot_learning_results.py: auxiliary functions plotting results
4. plot_results.ipynb: a jupyter notebook for loading and plotting results

You may want to edit the second file to change the parameters. You can do this with a text editor or IDE like IDLE or spyder (my python IDE of choice). Once you are satisfied, save the file.

If you are not using and IDE, you can run the file from the command line as follows: 

Step 1. Open the anaconda prompt and navigate to the desired folder using the cd command and the path:

Example:
cd "C:\Users\mpanaggio\Box\MRC\learn_model_fourier"

Step 2. Activate the environment with all of the necessary packages (if you created one). 

Example:
activate tf

Step 3. Run the file by typing:

python run_kuramoto_learn.py

Step 4. The results should be displayed in the command window.  They will also be saved to csv files (one for the frequencies, one for the matrix, and one for the coupling function).  You can view these results files with a text editor or spreadsheet software like microsoft excel.