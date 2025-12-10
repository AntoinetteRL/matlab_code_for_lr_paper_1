# Scalable-Hypergraph-Structure-Learning-with-Diverse-Smoothness-Priors

This repository contains code used for the paper **Scalable Hypergraph Structure Learning with Diverse Smoothness Priors** to generate the experimental results.
All code was ran using MATLAB version R2024b.

### **Setup Instructions**

In order to run the code provided, you need to first download and extract the zip file in a location of your choice. Then, you simply run the 'addPaths.m' function to automatically have all relevant code be on your system path. If you do not want these paths saved for future sessions, you can then run the 'clearPaths.m' function. Alternatively, you can comment out the line in 'addPaths.m' that saves the path changes and then the paths will be gone upon the start of your next session.

### **Running the Code**
The main folder of interest is the 'Experiments' folder. Each subfolder corresponds to one of the experimental results in the paper. Furthermore, some of them are split into the different methods used. For example, the 'Cora' experiment was performed with the method HSLS (our proposed method), HGSI, and one from Gao et al. (see paper).

Any of the source code can be ran on the corresponding dataset. They have names like 'Uniform_...' or 'Non_Uniform_...' and there are several parameters in the code that can be changed.

### **Recreation of Results**
To completely reconstruct the results from the paper, in each experimental subfolder find the 'Results' folder. There are saved MATLAB workspaces that can be loaded which contain all the variables used to get the reported results. If you want to test this yourself, you need to load the appropriate set of signal observations (often labeled something like 'X_v'), load the correct parameters (mostly stored under 'opts'), and then specify the correct total variaton hyperedge reduction method with appropriate number of nearest neighbors. Then, you can run the experimental code with keeping these variables constant (not overwritten by the code).
