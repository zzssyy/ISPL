# -*- coding: utf-8 -*-

import numpy as np
import CNN_feature
from collections import Counter
import re
import math
import platform
import os

def cnn_f(fastas):
    return CNN_feature.run(fastas)

def k_mers(rf, k):
    l = open(rf, 'r').readlines()
    ll = []
    j = 0
    for x in l:
        if len(x.strip()) != 0:
            if x[0] != '>':
                ll.append(x.strip().upper().replace('N', 'A').replace('M', 'A').replace('N', 'A').replace('W', 'A').replace('Y', 'T')
                          .replace('V', 'A').replace('H', 'A').replace('B', 'T').replace('D', 'A'))
            else:
                ll.append(str(j))
        j = j+1
    i = 0
    R = {}
    while i < len(ll):
        name = ll[i]
        var = ll[i + 1]
        R[name] = var
        i = i + 2

    veclen = 0
    for x in range(k):
        veclen = veclen + pow(4, k + 1)
    u = ['A', 'T', 'C', 'G']
    uu = []

    def quanpailie(k):
        if k == 1:
            for x in u:
                uu.append(x)
            return u
        else:
            gg = quanpailie(k - 1)
            temp = []
            for x in u:
                for y in gg:
                    uu.append(x + y)
                    temp.append(x + y)
            return temp

    quanpailie(k)
    for ee in R.keys():
        x = R[ee]
        d = {}
        for t in uu:
            d[t] = 0.0
        len1 = len(x)
        for i in range(k):
            i = i + 1
            s = len1 - i + 1
            w = 1.0 / (pow(4, k - i))
            for j in range(s):
                d[x[j:j + i]] = d[x[j:j + i]] + (1.0 * w / s)
        R[ee] = d.values()

    temp = []
    for x in R.keys():
        y = list(R[x])
        y.insert(0, x)
        temp.append(y)
    return temp

def checkFasta(fastas):
	status = True
	lenList = set()
	for i in fastas:
		lenList.add(len(i[1]))
	if len(lenList) == 1:
		return True
	else:
		return False

def minSequenceLength(fastas):
	minLen = 10000
	for i in fastas:
		if minLen > len(i[1]):
			minLen = len(i[1])
	return minLen

def minSequenceLengthWithNormalAA(fastas):
	minLen = 10000
	for i in fastas:
		if minLen > len(re.sub('-', '', i[1])):
			minLen = len(re.sub('-', '', i[1]))
	return minLen

def APAAC(fastas, lambdaValue=10, w=0.05):
	if minSequenceLengthWithNormalAA(fastas) < lambdaValue + 1:
		print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
		return 0

	dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\data\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records) - 1):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
		AAProperty1.append([(j - meanI) / fenmu for j in i])

	encodings = []
	header = ['#']
	for i in AA:
		header.append('Pc1.' + i)
	for j in range(1, lambdaValue + 1):
		for i in AAPropertyNames:
			header.append('Pc2.' + i + '.' + str(j))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		theta = []
		for n in range(1, lambdaValue + 1):
			for j in range(len(AAProperty1)):
				theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
								  range(len(sequence) - n)]) / (len(sequence) - n))
		myDict = {}
		for aa in AA:
			myDict[aa] = sequence.count(aa)

		code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
		code = code + [w * value / (1 + w * sum(theta)) for value in theta]
		encodings.append(code)
	return encodings

def CTDC_Count(seq1, seq2):
	sum = 0
	for aa in seq1:
		sum = sum + seq2.count(aa)
	return sum

def CTDC(fastas):
	group1 = {
		'hydrophobicity': 'RKEDQN',
		'surfacetension': 'GQDNAHR',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity': 'GASTPHY',
		'surfacetension': 'KTSEC',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity': 'CLVIMFW',
		'surfacetension': 'ILMFPWYV',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity', 'surfacetension', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	encodings = []
	header = ['#']
	for p in property:
		for g in range(1, len(groups) + 1):
			header.append(p + '.G' + str(g))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		for p in property:
			c1 = CTDC_Count(group1[p], sequence) / len(sequence)
			c2 = CTDC_Count(group2[p], sequence) / len(sequence)
			c3 = 1 - c1 - c2
			code = code + [c1, c2, c3]
		encodings.append(code)
	return encodings

def CTDT(fastas):
	group1 = {
		'hydrophobicity': 'RKEDQN',
		'surfacetension': 'GQDNAHR',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity': 'GASTPHY',
		'surfacetension': 'KTSEC',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity': 'CLVIMFW',
		'surfacetension': 'ILMFPWYV',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity', 'surfacetension', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	encodings = []
	header = ['#']
	for p in property:
		for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
			header.append(p + '.' + tr)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
		for p in property:
			c1221, c1331, c2332 = 0, 0, 0
			for pair in aaPair:
				if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
					c1221 = c1221 + 1
					continue
				if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
					c1331 = c1331 + 1
					continue
				if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
					c2332 = c2332 + 1
			code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
		encodings.append(code)
	return encodings

def CTDD_Count(aaSet, sequence):
	number = 0
	for aa in sequence:
		if aa in aaSet:
			number = number + 1
	cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
	cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

	code = []
	for cutoff in cutoffNums:
		myCount = 0
		for i in range(len(sequence)):
			if sequence[i] in aaSet:
				myCount += 1
				if myCount == cutoff:
					code.append((i + 1) / len(sequence) * 100)
					break
		if myCount == 0:
			code.append(0)
	return code


def CTDD(fastas):
	group1 = {
		'hydrophobicity': 'RKEDQN',
		'surfacetension': 'GQDNAHR',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity': 'GASTPHY',
		'surfacetension': 'KTSEC',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity': 'CLVIMFW',
		'surfacetension': 'ILMFPWYV',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity', 'surfacetension', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	encodings = []
	header = ['#']
	for p in property:
		for g in ('1', '2', '3'):
			for d in ['0', '25', '50', '75', '100']:
				header.append(p + '.' + g + '.residue' + d)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		for p in property:
			code = code + CTDD_Count(group1[p], sequence) + CTDD_Count(group2[p], sequence) + CTDD_Count(group3[p], sequence)
		encodings.append(code)
	return encodings

def AAC(fastas):
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

def read_aas(rf):
    seq = list()
    ids = list()

    with open(rf, "r") as lines:
        for data in lines:
            line = data.strip()
            if line[0] == '>':
                ids.append(line[1:])
            else:
                seq.append(line)
    datas = [[x, y] for x, y in zip(ids, seq)]

    return datas

def feature_integration(fastas, rf1, rf2):
	cnnf = cnn_f(rf1)
	kmers = k_mers(rf2, k=3)
	AC = AAC(fastas)
	AP = APAAC(fastas)
	CC = CTDC(fastas)
	CT = CTDT(fastas)
	CD = CTDD(fastas)
	ACTD = np.column_stack((np.array(AC), np.array(CC)[:, 1:], np.array(CT)[:, 1:], np.array(CD)[:, 1:]))
	seq = np.column_stack((np.array(cnnf), np.array(kmers)[:, 21:]))
	print(seq.shape)
	phy = np.column_stack((ACTD, np.array(AP)[:, 21:]))
	print(phy.shape)
	col = list()
	for i in range(len(seq[0])):
		col.append(str(i))
	seq = np.row_stack((np.array(col), seq))
	return phy, seq

def run(rf1, rf2):
	fastas_aas = read_aas(rf1)
	phy, seq = feature_integration(fastas_aas, rf1, rf2)
	return phy, seq
