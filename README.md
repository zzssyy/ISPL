# ISPL

An bioinformatics tool for identifying plant lncRNA-encoded short peptides

CDS and NCDS of Arabidopsis thaliana (A.thaliana), Glycine max (G.max), Zea mays (Z.mays) and Physcomitrella patens (P.patens) are downloaded from Phytozome v12 
(https://phytozome.jgi.doe.gov/pz/portal.html) website, including TAIR 10 (A.thaliana), release 189 (G.max), release PH207_443_v1.1 (Z.mays) and release 318_v3.3 (P.patens).
Sequences which are shorter than 303bp are selected. The above sequence redundancies are removed by using CD-HIT at the threshold of 0.8.

# Dependencies:
Language dependency: Python 3 (Please do not use Python 2 to run the code.)

# Library dependency:

Numpy 1.18.5

Keras 2.2.4

tensorflow 2.3.0

mlxtend 0.18.0

scikit-learn 0.22

# Usage

python ISPL.py input_fasta_file

input is the file of your data

# Demo

There is a sample sequence file sample_seqs.fa under the directory ./demo_files/. You can try to run ISPL on this file:

python ISPL.py ./demo_files/sample_seqs.fa


![image](https://github.com/zzssyy/bioinformatics/blob/master/Graphical-abstract.jpg)
