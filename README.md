# ISPL

An bioinformatics tool for identifying plant lncRNA-encoded short peptides

# OS
win10

# Dependencies
Language dependency: Python 3 (Please do not use Python 2 to run the code.)

# Library dependency

Numpy 1.18.5

Keras 2.2.4

tensorflow 2.3.0

mlxtend 0.18.0

scikit-learn 0.22

# Usage
There is a sample sequence file sample_seqs.fa under the directory ./demo_files/.

Our method includes two operation modes. When flag ='0', the number of positive and negative samples should be input for identification mode. You can try to run ISPL on this file:

python ISPL.py ./demo_files/sample_aas.fa ./demo_files/sample_RNAs.fa number_of_positive_samples number_of_negative_samples


When flag ='1', it is the prediction mode, and it is not necessary to input the number of positive and negative samples.You can try to run ISPL on this file:

python ISPL.py ./demo_files/sample_aas.fa ./demo_files/sample_RNAs.fa


# Running time

ILSP runs about 8s on the independent test test.

# Graphical abstract

![image](https://github.com/zzssyy/bioinformatics/blob/master/Graphical-abstract.png)
