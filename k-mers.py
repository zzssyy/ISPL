"""
提取序列k-mers特征
"""
import csv

def k_mers(rf, wf, k):
    l = open(rf, 'r').readlines()
    csvWriter = csv.writer(open(wf, 'w'))
    #
    ll = []
    for x in l:
        if len(x.strip()) != 0:
            if x[0] != '>':
                ll.append(x.strip().upper().replace('N', 'A').replace('M', 'A').replace('N', 'A').replace('W', 'A').replace('Y', 'T')
                          .replace('V', 'A').replace('H', 'A').replace('B', 'T').replace('D', 'A'))
            else:
                ll.append(x.strip()[1:].upper())
    # ll.pop(0)

    i = 0
    R = {}
    while i < len(ll):
#        name = ll[i].split(' ')[0]
        name = ll[i]
#        print(name)
        var = ll[i + 1]
        R[name] = var
        i = i + 2

    # print(R)

    veclen = 0
    for x in range(k):
        veclen = veclen + pow(4, k + 1)

    # 先创造出一个字典
    u = ['A', 'T', 'C', 'G']
    uu = []

    # for i in range(k):
    #     #m=[None]*4
    #     while i>=0:
    #         m = [''] * 4
    #         for j in len(m):
    #             m[j]=m[j]+u[j]

    # 写个递归函数
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
#    print(uu)
#    print(len(uu))

    for ee in R.keys():
        x = R[ee]
        d = {}
        for t in uu:
            d[t] = 0.0
        len1 = len(x)
        for i in range(k):
            i = i + 1
            s = len1 - i + 1  # 总的匹配次数
            w = 1.0 / (pow(4, k - i))
            for j in range(s):
                d[x[j:j + i]] = d[x[j:j + i]] + (1.0 * w / s)
        R[ee] = d.values()
    # D=d.values()
    # print(D)
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
#    rf = "G:\miRNA-encoded peptides\数据库\ARA-PEPs\k-mers结果\\SIP_ORFs(lncRNA).fa"
#    wf = "G:\miRNA-encoded peptides\数据库\ARA-PEPs\k-mers结果\\SIP_ORFs_lncRNA.fa"
    rf = ""
    wf = ""
    k = 4
    k_mers(rf, wf, k)



