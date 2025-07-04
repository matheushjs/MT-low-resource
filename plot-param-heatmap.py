import numpy as np
import matplotlib.pyplot as plt

text = """
lambda = 0.333	exponent = 0.333	avg. correlation = -0.3562406142778194
lambda = 0.333	exponent = 0.500	avg. correlation = -0.349403757310814
lambda = 0.333	exponent = 0.667	avg. correlation = -0.32671851401821994
lambda = 0.333	exponent = 0.833	avg. correlation = -0.3075839834503176
lambda = 0.333	exponent = 1.000	avg. correlation = -0.2888867359127631
lambda = 0.333	exponent = 1.200	avg. correlation = -0.2685464519932278
lambda = 0.333	exponent = 1.500	avg. correlation = -0.23441527968648318
lambda = 0.333	exponent = 2.000	avg. correlation = -0.18811152370004885
lambda = 0.333	exponent = 3.000	avg. correlation = -0.14392480351859815
lambda = 0.500	exponent = 0.333	avg. correlation = -0.3373461235989753
lambda = 0.500	exponent = 0.500	avg. correlation = -0.36348133764170165
lambda = 0.500	exponent = 0.667	avg. correlation = -0.3753693311043461
lambda = 0.500	exponent = 0.833	avg. correlation = -0.36559162921492866
lambda = 0.500	exponent = 1.000	avg. correlation = -0.35105996250143107
lambda = 0.500	exponent = 1.200	avg. correlation = -0.33945097349440295
lambda = 0.500	exponent = 1.500	avg. correlation = -0.3187238948193844
lambda = 0.500	exponent = 2.000	avg. correlation = -0.27872764810276773
lambda = 0.500	exponent = 3.000	avg. correlation = -0.20801564930660543
lambda = 0.667	exponent = 0.333	avg. correlation = -0.32818837542792584
lambda = 0.667	exponent = 0.500	avg. correlation = -0.34534913914715915
lambda = 0.667	exponent = 0.667	avg. correlation = -0.36354584001100393
lambda = 0.667	exponent = 0.833	avg. correlation = -0.37239044277613353
lambda = 0.667	exponent = 1.000	avg. correlation = -0.37478351440185487
lambda = 0.667	exponent = 1.200	avg. correlation = -0.3663264085156714
lambda = 0.667	exponent = 1.500	avg. correlation = -0.35486653234570786
lambda = 0.667	exponent = 2.000	avg. correlation = -0.32518352602328243
lambda = 0.667	exponent = 3.000	avg. correlation = -0.2682540493752251
lambda = 0.833	exponent = 0.333	avg. correlation = -0.3352224396595827
lambda = 0.833	exponent = 0.500	avg. correlation = -0.34652858444101586
lambda = 0.833	exponent = 0.667	avg. correlation = -0.35540544943841107
lambda = 0.833	exponent = 0.833	avg. correlation = -0.3641768157599877
lambda = 0.833	exponent = 1.000	avg. correlation = -0.3698192725822851
lambda = 0.833	exponent = 1.200	avg. correlation = -0.3692660998842786
lambda = 0.833	exponent = 1.500	avg. correlation = -0.3695926689054377
lambda = 0.833	exponent = 2.000	avg. correlation = -0.3521301665859957
lambda = 0.833	exponent = 3.000	avg. correlation = -0.3075544611070845
lambda = 1.000	exponent = 0.333	avg. correlation = -0.33779761475006387
lambda = 1.000	exponent = 0.500	avg. correlation = -0.3539156594628461
lambda = 1.000	exponent = 0.667	avg. correlation = -0.35879108669608517
lambda = 1.000	exponent = 0.833	avg. correlation = -0.363285355613807
lambda = 1.000	exponent = 1.000	avg. correlation = -0.3660339553097004
lambda = 1.000	exponent = 1.200	avg. correlation = -0.3672060934813387
lambda = 1.000	exponent = 1.500	avg. correlation = -0.36884868583941394
lambda = 1.000	exponent = 2.000	avg. correlation = -0.3666145437000027
lambda = 1.000	exponent = 3.000	avg. correlation = -0.3366649752651971
lambda = 1.200	exponent = 0.333	avg. correlation = -0.3416689126790994
lambda = 1.200	exponent = 0.500	avg. correlation = -0.3593434152292406
lambda = 1.200	exponent = 0.667	avg. correlation = -0.3674821815103985
lambda = 1.200	exponent = 0.833	avg. correlation = -0.36790556972851474
lambda = 1.200	exponent = 1.000	avg. correlation = -0.36749659979677335
lambda = 1.200	exponent = 1.200	avg. correlation = -0.3676486382789423
lambda = 1.200	exponent = 1.500	avg. correlation = -0.3610027044638236
lambda = 1.200	exponent = 2.000	avg. correlation = -0.37089099062920283
lambda = 1.200	exponent = 3.000	avg. correlation = -0.3589564247089883
lambda = 1.500	exponent = 0.333	avg. correlation = -0.34030155175840365
lambda = 1.500	exponent = 0.500	avg. correlation = -0.36584737901801656
lambda = 1.500	exponent = 0.667	avg. correlation = -0.3739513778330196
lambda = 1.500	exponent = 0.833	avg. correlation = -0.37718115770566174
lambda = 1.500	exponent = 1.000	avg. correlation = -0.37705613272415406
lambda = 1.500	exponent = 1.200	avg. correlation = -0.37279940911830656
lambda = 1.500	exponent = 1.500	avg. correlation = -0.3660770818844583
lambda = 1.500	exponent = 2.000	avg. correlation = -0.36755523141405816
lambda = 1.500	exponent = 3.000	avg. correlation = -0.3684369487263419
lambda = 2.000	exponent = 0.333	avg. correlation = -0.32952354110528953
lambda = 2.000	exponent = 0.500	avg. correlation = -0.3607191321057121
lambda = 2.000	exponent = 0.667	avg. correlation = -0.382274843349489
lambda = 2.000	exponent = 0.833	avg. correlation = -0.38564329196364466
lambda = 2.000	exponent = 1.000	avg. correlation = -0.3863604668451583
lambda = 2.000	exponent = 1.200	avg. correlation = -0.38428771503271664
lambda = 2.000	exponent = 1.500	avg. correlation = -0.37937406424827835
lambda = 2.000	exponent = 2.000	avg. correlation = -0.3689754510794604
lambda = 2.000	exponent = 3.000	avg. correlation = -0.37070689149662633
lambda = 3.000	exponent = 0.333	avg. correlation = -0.31542513773678327
lambda = 3.000	exponent = 0.500	avg. correlation = -0.3481977044134135
lambda = 3.000	exponent = 0.667	avg. correlation = -0.3766539469851624
lambda = 3.000	exponent = 0.833	avg. correlation = -0.3927447516872482
lambda = 3.000	exponent = 1.000	avg. correlation = -0.396411171675173
lambda = 3.000	exponent = 1.200	avg. correlation = -0.3962798005210763
lambda = 3.000	exponent = 1.500	avg. correlation = -0.3934745866669222
lambda = 3.000	exponent = 2.000	avg. correlation = -0.3839856914112928
lambda = 3.000	exponent = 3.000	avg. correlation = -0.3818875091086811
lambda = 3.500	exponent = 0.333	avg. correlation = -0.3127777066838332
lambda = 3.500	exponent = 0.500	avg. correlation = -0.3448734958437575
lambda = 3.500	exponent = 0.667	avg. correlation = -0.3724181061178639
lambda = 3.500	exponent = 0.833	avg. correlation = -0.39247104856671555
lambda = 3.500	exponent = 1.000	avg. correlation = -0.3992793840141119
lambda = 3.500	exponent = 1.200	avg. correlation = -0.3989694576323005
lambda = 3.500	exponent = 1.500	avg. correlation = -0.39729220709549484
lambda = 3.500	exponent = 2.000	avg. correlation = -0.3922574602266709
lambda = 3.500	exponent = 3.000	avg. correlation = -0.3878389291847023
lambda = 4.000	exponent = 0.333	avg. correlation = -0.17318911292332345
lambda = 4.000	exponent = 0.500	avg. correlation = -0.3424783439095537
lambda = 4.000	exponent = 0.667	avg. correlation = -0.36955383591598984
lambda = 4.000	exponent = 0.833	avg. correlation = -0.39090846693771353
lambda = 4.000	exponent = 1.000	avg. correlation = -0.4003696783664877
lambda = 4.000	exponent = 1.200	avg. correlation = -0.40119313394528877
lambda = 4.000	exponent = 1.500	avg. correlation = -0.4007425970763627
lambda = 4.000	exponent = 2.000	avg. correlation = -0.3975131768865076
lambda = 4.000	exponent = 3.000	avg. correlation = -0.3933917386378808
lambda = 4.500	exponent = 0.333	avg. correlation = -0.12599157055622662
lambda = 4.500	exponent = 0.500	avg. correlation = -0.20344294645612598
lambda = 4.500	exponent = 0.667	avg. correlation = -0.3671383522800047
lambda = 4.500	exponent = 0.833	avg. correlation = -0.38864511540464913
lambda = 4.500	exponent = 1.000	avg. correlation = -0.40092482878926006
lambda = 4.500	exponent = 1.200	avg. correlation = -0.4029622640442711
lambda = 4.500	exponent = 1.500	avg. correlation = -0.40296696523485415
lambda = 4.500	exponent = 2.000	avg. correlation = -0.4013280142691295
lambda = 4.500	exponent = 3.000	avg. correlation = -0.39797624716909163
lambda = 5.000	exponent = 0.333	avg. correlation = -0.12442801837708775
lambda = 5.000	exponent = 0.500	avg. correlation = -0.1191755603764891
lambda = 5.000	exponent = 0.667	avg. correlation = -0.22817429387602797
lambda = 5.000	exponent = 0.833	avg. correlation = -0.38669416800415357
lambda = 5.000	exponent = 1.000	avg. correlation = -0.40075875587972476
lambda = 5.000	exponent = 1.200	avg. correlation = -0.40426043601207395
lambda = 5.000	exponent = 1.500	avg. correlation = -0.40428923877468925
lambda = 5.000	exponent = 2.000	avg. correlation = -0.4038376560638685
lambda = 5.000	exponent = 3.000	avg. correlation = -0.40161102786107594
lambda = 6.000	exponent = 0.333	avg. correlation = -0.126824614848209
lambda = 6.000	exponent = 0.500	avg. correlation = -0.1492852134356074
lambda = 6.000	exponent = 0.667	avg. correlation = -0.170299008762006
lambda = 6.000	exponent = 0.833	avg. correlation = -0.16436142412362628
lambda = 6.000	exponent = 1.000	avg. correlation = -0.2623242677008374
lambda = 6.000	exponent = 1.200	avg. correlation = -0.4057189436684758
lambda = 6.000	exponent = 1.500	avg. correlation = -0.40544252670905356
lambda = 6.000	exponent = 2.000	avg. correlation = -0.40671881033764895
lambda = 6.000	exponent = 3.000	avg. correlation = -0.40663627463868146
lambda = 7.500	exponent = 0.333	avg. correlation = 0.0
lambda = 7.500	exponent = 0.500	avg. correlation = 0.0
lambda = 7.500	exponent = 0.667	avg. correlation = -0.14220586347998823
lambda = 7.500	exponent = 0.833	avg. correlation = -0.18262858422631106
lambda = 7.500	exponent = 1.000	avg. correlation = -0.19401136392024423
lambda = 7.500	exponent = 1.200	avg. correlation = -0.18776211874136178
lambda = 7.500	exponent = 1.500	avg. correlation = -0.4068583396846962
lambda = 7.500	exponent = 2.000	avg. correlation = -0.4081785433324133
lambda = 7.500	exponent = 3.000	avg. correlation = -0.4107708283806413
lambda = 10.000	exponent = 0.333	avg. correlation = 0.0
lambda = 10.000	exponent = 0.500	avg. correlation = 0.0
lambda = 10.000	exponent = 0.667	avg. correlation = 0.0
lambda = 10.000	exponent = 0.833	avg. correlation = 0.0
lambda = 10.000	exponent = 1.000	avg. correlation = 0.0
lambda = 10.000	exponent = 1.200	avg. correlation = -0.15444075410900687
lambda = 10.000	exponent = 1.500	avg. correlation = -0.19765494366787667
lambda = 10.000	exponent = 2.000	avg. correlation = -0.1902971280260975
lambda = 10.000	exponent = 3.000	avg. correlation = -0.4128708185494765
"""

data = [ [ float(j.split(" ")[-1]) for j in i.split("\t") ] for i in  text.strip().split("\n") ]

#data = [ [ float(j) for j in i.split(" ") ] for i in open("output.txt").read().strip().split("\n") ]

mat = np.array(data)[:,-1].reshape(-1, 9)

xticklabs = "1/3, 1/2, 1/1.5, 1/1.2, 1, 1.2, 1.5, 2, 3".split(", ")
yticklabs = "1/3, 1/2, 1/1.5, 1/1.2, 1, 1.2, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 7.5, 10".split(", ")

mat = mat[::-1,:]
yticklabs = yticklabs[::-1]

plt.imshow(mat)
cbar = plt.colorbar()
#cbar.set_label("Correlation", rotation=270)
plt.xlabel("Exponent ($\\tau$)")
plt.ylabel("Lambda ($\\lambda$)")
plt.xticks(ticks=list(range(len(xticklabs))), labels=xticklabs)
plt.yticks(ticks=list(range(len(yticklabs))), labels=yticklabs)
plt.tight_layout()

filename = "heatmap-cltad-params.png"
try:
    open(filename)
    print("File exists. Will not save.")
except:
    print(f"Saving plot to {filename}.")
    plt.savefig(filename, dpi=150)

plt.show()