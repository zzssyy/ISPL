# bioinformatics

An bioinformatics tool for identifying plant lncRNA-encoded short peptides

CDS and NCDS of Arabidopsis thaliana (A.thaliana), Glycine max (G.max), Zea mays (Z.mays) and Physcomitrella patens (P.patens) are downloaded from Phytozome v12 
(https://phytozome.jgi.doe.gov/pz/portal.html) website, including TAIR 10 (A.thaliana), release 189 (G.max), release PH207_443_v1.1 (Z.mays) and release 318_v3.3 (P.patens).
Sequences which are shorter than 303bp are selected. The above sequence redundancies are removed by using CD-HIT at the threshold of 0.8.

1.GffParser.py

Sequence format conversion

"""

usage:

to extract introns:

GP = GffParser("Ppatens_318_v3.3.gene_exons.gff3")

GP.writeIntrons("Ppatens_318_v3.3_introns.bed")

"""

2.k-mers.py

Get the k-mer feature

"""

usage:

rf: the input path; wf: the output path; k: k value of k-mer

"""

3.wordvectrain.py

Train word vector

4.proprecessing.py

Data preprocessing

5.model.py

Train the CNN model

6.CNN_feature.py

Local features based on CNN adaptive extraction

7.feature_engineering.py

Used to fuse different modal features

"""

usage:

path11: the input path of k-mer (positive samples); path12: the input path of k-mer (negative samples)

path2: the input path of cnn-f; 

path31: the input path of APAAC (positive samples); path32: the input path of APAAC (negative samples)

path41: the input path of 188D (positive samples); path42: the input path of 188D (negative samples)

wf12: the output of sequence feature; wf34: the output of physicochemical feature

"""

8.IMRMD.py

IMRMD feature selection

"""

usage:

python3 IMRMD.py -i input.csv -o matix.csv -c output.csv

"""

9.stacking.py

Ensemble learning stacking method to obtain the final result

"""

usage:

path1: the input path of sequence feature after feature selection; path2: the input path of physicochemical feature after feature selection

"""

You can also use ISPL.py to test your data

usage:

python ISPL.py path

path is the path your data

![image](https://github.com/zzssyy/bioinformatics/blob/master/Graphical-abstract.jpg)
