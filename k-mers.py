import csv
import sys
def k_mers(rf, wf, k):
    l = open(rf, 'r').readlines()
    csvWriter = csv.writer(open(wf, 'w'))
    ll = []
    for x in l:
        if len(x.strip()) != 0:
            if x[0] != '>':
                ll.append(x.strip().upper().replace('N', 'A').replace('M', 'A').replace('N', 'A').replace('W', 'A').replace('Y', 'T')
                          .replace('V', 'A').replace('H', 'A').replace('B', 'T').replace('D', 'A'))
            else:
                ll.append(x.strip()[1:].upper())
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
    print(len(temp))
    with open(wf, 'w', newline='') as fout:
        cin = csv.writer(fout)
        cin.writerows(temp)


if __name__ == "__main__":
    pass
    rf = sys.argv[1]
    wf = sys.argv[2]
    k = 4
    k_mers(rf, wf, k)



