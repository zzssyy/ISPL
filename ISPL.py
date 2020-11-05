import os
import sys

path = sys.argv[1]

#CNN feature
os.system("python CNN_feature.py path+ '/ '+ pos.fa path+ '/ '+ neg.fa path+ '/ '+ cnn_f.tsv")
#feature fusion
os.system("python feature_engineering.py path+ '/ '+ kmer_pos.tsv path+ '/ '+ kmer_neg.tsv path+ '/ '+ cnn_f.tsv path+ '/ '+ 188d_pos.tcv path+ '/ '+ 188d_neg.tsv path+ '/ '+ apaac_pos.tsv path+ '/ '+ apaac_neg.tsv path+ '/ '+ cnn-mer.tsv path+ '/ '+ 188-AC.tsv")
#feature selection
os.system("python reduce_dimension.py path+ '/ '+ cnn-mer.tsv path+ '/ '+ 188-AC.tsv path+ '/ '+ cnn-mer'.tsv path+ '/ '+ 188-AC.tsv")
#ensemble learning
os.system("python CNN_feature.py path+ '/ '+ path+ '/ '+ cnn-mer'.tsv path+ '/ '+ 188-AC'.tsv")