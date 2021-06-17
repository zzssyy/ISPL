import sys
import feature_engineering
import reduce_dimension
import stacking

# imput_aas = "C:\\Users\Wo1verien\Desktop\\bioinformatics-master\sample\\test_aa.fa"
# input_RNAs = "C:\\Users\Wo1verien\Desktop\\bioinformatics-master\sample\\test_RNA.fa"

input_aas = sys.argv[0]
input_RNAs = sys.argv[1]
flag = sys.argv[2]
m = sys.argv[3]
n = sys.argv[4]

#feature fusion
phy, seq = feature_engineering.run(input_aas, input_RNAs)

#feature selection
seq, phy = reduce_dimension.run(seq, phy)

#ensemble learning
if flag == '0':
    stacking.identify(seq, phy, m, n)
else:
    stacking.predict(seq, phy)