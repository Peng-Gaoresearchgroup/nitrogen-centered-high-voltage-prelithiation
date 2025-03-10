# import pandas as pd
# import re
# # print(elemental_statistics(pd.read_csv('./data/onlyNLi.csv')))

# df=pd.read_csv('./data/onlyNLi.csv')['canonicalsmiles'].to_list()
# df2=pd.read_csv('./data/onlyNH.csv')['canonicalsmiles'].to_list()

# print(df[396],df[44],df[45],df[46])
# for i in df:
#     if i == '[Li]N1c2ccccc2N([Li])C1S':
#         print(df.index(i)) 
# # print(df.index('[Li]n1c(=O)n([Li])c2ccccc21'))

# # # li=['cluster11_mol13_', 'cluster8_mol56_', 'cluster3_mol518_', 'cluster4_mol226_', 'cluster0_mol468_', 'cluster7_mol342_', 'cluster5_mol264_', 'cluster9_mol35_', 'cluster9_mol196_', 'cluster2_mol490_', 'cluster6_mol180_', 'cluster10_mol145_', 'cluster1_mol554_']
# # # li.sort()
# # # print(li)

# # li=[32, 34, 44, 45, 46, 63, 100, 102, 104, 105, 111, 113, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 178, 179, 195, 212, 227, 247, 248, 258, 262, 267, 291, 307, 313, 331, 353, 354, 355, 372, 379, 380, 381, 382, 383, 396, 402]
# # print(len(li))
# # li.sort()
# # smiles=[df[i] for i in li if i !=35 and i  !=196]
# # print(len(smiles),'\n',smiles)
# # df=pd.DataFrame({'1':[1],'2':[5]})
# # df['3']=df['1']+df['2']
# # print(df)


a=1
b=a
a=2
print(b)