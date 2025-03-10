import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D
import pandas as pd
from typing import TypedDict
import re,time,json
import requests as rq
import numpy as np

#-----------------------------------------------------
# Modifying a dictionary to make indexing easier, dict['A']['B'] == c ----> dict.A.B == c
#-----------------------------------------------------

class SafeDict(dict):
    def __getattr__(self, name):
        value = self.get(name, None) 
        if isinstance(value, dict):
            return SafeDict(value)
        elif isinstance(value, list):
            return [SafeDict(item) if isinstance(item, dict) else item for item in value]
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __dir__(self):
        return super().__dir__() + list(self.keys())

#-----------------------------------------------------
# Data process and statistics
#-----------------------------------------------------
def get_valid_smiles(df):
    li=df['canonicalsmiles'].to_list()
    print(len(li))
    new=[]
    for i in li:
        if Chem.MolFromSmiles(i) != None:
            new.append(Chem.MolToSmiles(Chem.MolFromSmiles(i)))
    print(len(new))
    return pd.DataFrame({'canonicalsmiles':new})

def get_all_elements(df):
    li=df['canonicalsmiles'].to_list()
    element=set()
    for smiles in li:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol:
            for atom in mol.GetAtoms():
                element.add(atom.GetSymbol())
    return element

def get_elements_distribution(df,elements):
    li=df['canonicalsmiles'].to_list()
    count = {element: 0 for element in elements}
    for smiles in li:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol:
            atoms=[i.GetSymbol() for i in mol.GetAtoms()]
            for element in elements:
                if element in atoms:
                    count[element] += 1
    sorted_count=sorted(count.items(), key=lambda x: x[1], reverse=False)
    return sorted_count

def get_molWt_distribution(df):
    li=df['canonicalsmiles'].to_list()
    molWt=[]
    for smiles in li:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol:
            molWt.append(Descriptors.MolWt(mol))
    return molWt

#-----------------------------------------------------
# Smiles process
#-----------------------------------------------------
def check_NH(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is None: 
        return None
    total_h_count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N":
            total_h_count += atom.GetTotalNumHs()
    return total_h_count

def only_NH(smiles):
    mol = Chem.MolFromSmiles(smiles)
    li=[]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N":
            if atom.GetTotalNumHs() == 1:
                li.append(True)
            else:
                li.append(False)
    if False not in li:
        return True
    else:
        return False

def smiles2descirptors(smiles,desdic,seed): #desdic is a dic containing what descriptors we choose,{_get:['a','b'],_2d:['c']......}, see conf.yaml.
    '''Input a SMILES string, return its descriptors' pd.series calculated by rdkit'''
    print(f'Processing {smiles}')    
    mol=Chem.MolFromSmiles(smiles) 
    if mol is None:
        raise ValueError("SMILES Error")
    try:
        # Get 2D descriptors
        des={f'{i}': getattr(mol, f'Get{i}')() for i in desdic._get}
        des.update({f'{i}': getattr(Descriptors, f'{i}')(mol) for i in desdic._2d})
        # 3D Embed
        AllChem.EmbedMolecule(mol, useRandomCoords=True,maxAttempts=5000, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(mol)
        mol=Chem.AddHs(mol)
        params = AllChem.ETKDG()
        params.randomSeed = seed
        AllChem.EmbedMolecule(mol, params)
        # Get 3D descriptors
        des.update({f'{i}': getattr(Descriptors3D, f'{i}')(mol) for i in desdic._3d})
        print(f'{smiles} Processing done')
        return pd.Series(des)
    except:
        print(f'{smiles} Error')
        des={f'{i}': None for i in desdic._get}
        des.update({f'{i}': None for i in desdic._2d})
        des.update({f'{i}': None for i in desdic._3d})
        return(pd.Series(des))

def extract_xy(string):
    pattern = r'\[\s?([\d\.\-eE]+)\s+([\d\.\-eE]+)\s?\s?\s?\]'
    match = re.match(pattern, string)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return x, y
    else:
        # raise ValueError(f"无法从字符串中提取 x 和 y: {string}")
        return None,None

def NH2NLi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  
    
    pattern = Chem.MolFromSmarts("[#7]~[#1]")
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))  
    
    h_indices = sorted({m[1] for m in matches}, reverse=True)
    emol = Chem.EditableMol(mol)
    for idx in h_indices:
        emol.ReplaceAtom(idx, Chem.Atom(3))  
        
    new_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except:
        pass
    new_mol = Chem.RemoveHs(new_mol)
    return Chem.MolToSmiles(new_mol)

#-----------------------------------------------------
# Molecule Ranking
#-----------------------------------------------------

def get_scscore(smiles):
    time.sleep(2)
    api_url = 'https://askcos.mit.edu/api/molecular-complexity/call-async'
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0',
             "Accept": "application/json",
            "Content-Type": "application/json"
            }
    request_data = {
        "smiles": [smiles],
        "complexity_metrics": ["scscore"]
    }
    print(f'SCScore: Processing {smiles}')
    re1 = rq.post(api_url, json=request_data,headers=headers)
    if re1.status_code == 200:
        re2=rq.get(f'https://askcos.mit.edu/api/legacy/celery/task/{re1.json()}')
        if re2.status_code == 200:
            print(f"Post sucess: {re2.status_code}")
            data=json.loads(re2.text)
            state = data["state"]
            result =data["output"]["result"]
            if state == "SUCCESS":
                print(f"Get sucess: {re2.status_code}")
                return 5-float(result[0]["scscore"])
            else:
                print(f"Get failed: {re2.status_code}")
                return None
    else:
        print(f"Post failed: {re1.status_code}")
        return None

def get_specific_capacity(smiles):
        mol=Chem.MolFromSmiles(smiles)
        mol=Chem.AddHs(mol)
        mol_weight = Descriptors.MolWt(mol)
        li_num = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Li')
        specific_capacity=li_num*96500/(3.6*mol_weight)
        return specific_capacity

def get_element_freindless(smiles,elemnts):
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    count={}
    rank_list={'N': 1, 'C': 0, 'Li': 0, 'O': 0, 'Mg': -1, 'Cl': -1, 'Br': -1, 'F': 1, 'P': 0, 'Na': -1, 'B': 0, 'Si': 0, 'H': 0, 'S': 0}
    for element in  elemnts:
        num = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)/mol.GetNumAtoms()
        count[element]=num
    rank=sum([rank_list[i]*count[i] for i in elemnts])
    return count,rank

#-----------------------------------------------------
# Data analyze
#-----------------------------------------------------

def get_representative_mol(df,save):
    result = df.loc[df.groupby('Cluster')['Distance_to_center'].idxmin(), ['Molecule','Cluster','data','Cluster_center','Distance_to_center']]
    result.to_csv(save,index=False)
    return result

def get_confusion_matrix(dict_hc=None, dict_kmeans=None):
    if dict_hc ==None:
        # dict_hc={'Cluster0': [0, 2, 3, 6, 8, 17, 36, 47, 57, 71, 75, 168, 184, 197, 198, 201, 207, 215, 217, 238, 250, 251, 260, 274, 285, 290, 334, 337, 340, 342, 368, 388, 391, 404, 502, 547, 548, 555, 558, 559, 560, 564],
        #         'Cluster1': [1, 10, 14, 15, 16, 22, 24, 25, 26, 27, 30, 33, 40, 41, 49, 50, 60, 77, 83, 85, 86, 88, 91, 97, 98, 99, 101, 108, 112, 116, 120, 121, 135, 137, 139, 142, 151, 170, 174, 187, 188, 192, 193, 204, 205, 210, 226, 233, 244, 256, 257, 276, 279, 282, 283, 284, 320, 332, 335, 346, 348, 349, 384, 506, 507, 574, 575, 577, 578, 579, 580, 581],
        #         'Cluster2': [4, 11, 13, 66, 109, 124, 131, 141, 150, 373],
        #         'Cluster3': [5, 115, 123, 126, 222, 252, 253, 263, 264, 277, 278, 287, 292, 294, 301, 308, 309, 311, 315, 318, 327, 328, 329, 343, 359, 362, 371, 374, 385, 563, 565, 566, 582, 583],
        #         'Cluster4': [7, 19, 37, 48, 52, 72, 76, 78, 143, 169, 218, 232, 255, 286, 304, 305, 306, 314, 347, 356, 364, 367, 370, 389, 390, 392, 408, 419, 422, 424, 426, 427, 428, 432, 433, 440, 441, 442, 443, 450, 452, 455, 456, 462, 463, 464, 471, 473, 474, 475, 476, 491, 492, 517, 518, 521, 527, 528, 529, 530, 532, 533, 534, 535, 537, 542, 543, 567, 569, 572, 573],
        #         'Cluster5': [9, 56, 59, 106, 114, 119, 125, 128, 165, 185, 200, 202, 214, 239, 249, 268, 275, 280, 281, 295, 296, 321, 323, 324, 326, 360, 361, 366, 386, 387, 397, 399, 470, 503, 553, 556, 557],
        #         'Cluster6': [12, 21, 23, 38, 53, 61, 79, 81, 82, 87, 89, 93, 118, 132, 133, 136, 144, 147, 148, 172, 180, 181, 189, 190, 191, 211, 213, 216, 224, 225, 242, 266, 270, 271, 272, 273, 289, 297, 303, 310, 316, 325, 344, 352, 405, 444, 445, 446, 447, 448, 453, 454, 457, 458, 459, 460, 495, 497, 504, 508, 509, 510, 511, 522, 545],
        #         'Cluster7': [18, 20, 39, 51, 58, 65, 67, 73, 84, 107, 117, 130, 138, 203, 236, 243, 254, 261, 317, 341, 365, 369, 398, 410, 434, 436, 449, 451, 461, 489, 490, 493, 494, 496, 498, 499, 505, 536, 540, 544, 549, 550, 570, 571, 576],
        #         'Cluster8': [28, 29, 42, 43, 54, 55, 80, 90, 94, 95, 96, 110, 122, 149, 175, 176, 177, 182, 183, 245, 246, 293, 312, 330, 345, 350, 351, 375, 376, 377, 395, 465, 466, 467, 477, 523, 538, 546, 551, 552, 554, 584],
        #         'Cluster9': [31, 32, 34, 35, 44, 45, 46, 62, 63, 100, 102, 103, 104, 105, 111, 113, 134, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 178, 179, 195, 196, 206, 212, 227, 235, 247, 248, 258, 262, 267, 291, 302, 307, 313, 331, 353, 354, 355, 372, 378, 379, 380, 381, 382, 383, 393, 396, 401, 402],
        #         'Cluster10': [64, 68, 69, 70, 74, 127, 129, 140, 164, 166, 167, 186, 199, 208, 209, 219, 220, 221, 237, 240, 241, 259, 319, 333, 336, 338, 339, 357, 358, 363, 403, 406, 407, 409, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421, 423, 425, 429, 430, 431, 435, 437, 438, 439, 468, 469, 472, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 500, 501, 512, 513, 514, 515, 516, 519, 520, 524, 525, 526, 531, 539, 541, 561, 562, 568],
        #         'Cluster11': [92, 145, 146, 152, 171, 173, 194, 223, 228, 229, 230, 231, 234, 265, 269, 288, 298, 299, 300, 322, 394, 400]
        #         }
        dict_hc={'Cluster0': [64, 68, 69, 70, 74, 127, 129, 140, 164, 166, 167, 186, 199, 208, 209, 219, 220, 221, 237, 240, 241, 259, 319, 333, 336, 338, 339, 357, 358, 363, 403, 406, 407, 409, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421, 423, 425, 429, 430, 431, 435, 437, 438, 439, 468, 469, 472, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 500, 501, 512, 513, 514, 515, 516, 519, 520, 524, 525, 526, 531, 539, 541, 561, 562, 568],
                'Cluster1': [28, 29, 42, 43, 54, 55, 80, 90, 94, 95, 96, 110, 122, 149, 175, 176, 177, 182, 183, 245, 246, 293, 312, 330, 345, 350, 351, 375, 376, 377, 395, 465, 466, 467, 477, 523, 538, 546, 551, 552, 554, 584],
                'Cluster2': [18, 20, 39, 51, 58, 65, 67, 73, 84, 107, 117, 130, 138, 203, 236, 243, 254, 261, 317, 341, 365, 369, 398, 410, 434, 436, 449, 451, 461, 489, 490, 493, 494, 496, 498, 499, 505, 536, 540, 544, 549, 550, 570, 571, 576],
                'Cluster3': [7, 19, 37, 48, 52, 72, 76, 78, 143, 169, 218, 232, 255, 286, 304, 305, 306, 314, 347, 356, 364, 367, 370, 389, 390, 392, 408, 419, 422, 424, 426, 427, 428, 432, 433, 440, 441, 442, 443, 450, 452, 455, 456, 462, 463, 464, 471, 473, 474, 475, 476, 491, 492, 517, 518, 521, 527, 528, 529, 530, 532, 533, 534, 535, 537, 542, 543, 567, 569, 572, 573],
                'Cluster4': [1, 10, 14, 15, 16, 22, 24, 25, 26, 27, 30, 33, 40, 41, 49, 50, 60, 77, 83, 85, 86, 88, 91, 97, 98, 99, 101, 108, 112, 116, 120, 121, 135, 137, 139, 142, 151, 170, 174, 187, 188, 192, 193, 204, 205, 210, 226, 233, 244, 256, 257, 276, 279, 282, 283, 284, 320, 332, 335, 346, 348, 349, 384, 506, 507, 574, 575, 577, 578, 579, 580, 581],
                'Cluster5': [5, 115, 123, 126, 222, 252, 253, 263, 264, 277, 278, 287, 292, 294, 301, 308, 309, 311, 315, 318, 327, 328, 329, 343, 359, 362, 371, 374, 385, 563, 565, 566, 582, 583],
                'Cluster6': [12, 21, 23, 38, 53, 61, 79, 81, 82, 87, 89, 93, 118, 132, 133, 136, 144, 147, 148, 172, 180, 181, 189, 190, 191, 211, 213, 216, 224, 225, 242, 266, 270, 271, 272, 273, 289, 297, 303, 310, 316, 325, 344, 352, 405, 444, 445, 446, 447, 448, 453, 454, 457, 458, 459, 460, 495, 497, 504, 508, 509, 510, 511, 522, 545],
                'Cluster7': [0, 2, 3, 6, 8, 17, 36, 47, 57, 71, 75, 168, 184, 197, 198, 201, 207, 215, 217, 238, 250, 251, 260, 274, 285, 290, 334, 337, 340, 342, 368, 388, 391, 404, 502, 547, 548, 555, 558, 559, 560, 564],
                'Cluster8': [9, 56, 59, 106, 114, 119, 125, 128, 165, 185, 200, 202, 214, 239, 249, 268, 275, 280, 281, 295, 296, 321, 323, 324, 326, 360, 361, 366, 386, 387, 397, 399, 470, 503, 553, 556, 557],
                'Cluster9': [31, 32, 34, 35, 44, 45, 46, 62, 63, 100, 102, 103, 104, 105, 111, 113, 134, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 178, 179, 195, 196, 206, 212, 227, 235, 247, 248, 258, 262, 267, 291, 302, 307, 313, 331, 353, 354, 355, 372, 378, 379, 380, 381, 382, 383, 393, 396, 401, 402],
                'Cluster10': [92, 145, 146, 152, 171, 173, 194, 223, 228, 229, 230, 231, 234, 265, 269, 288, 298, 299, 300, 322, 394, 400],
                'Cluster11': [4, 11, 13, 66, 109, 124, 131, 141, 150, 373]
                }
        # dict_hc = dict(sorted(dict_hc.items()))
    if dict_kmeans ==None:
        dict_kmeans={'Cluster0': [37, 64, 127, 129, 164, 166, 167, 169, 199, 208, 209, 219, 220, 221, 240, 241, 287, 315, 325, 333, 336, 356, 357, 358, 363, 367, 403, 406, 409, 411, 415, 416, 417, 418, 420, 421, 423, 425, 429, 430, 431, 438, 441, 442, 443, 468, 469, 472, 474, 475, 478, 479, 486, 488, 500, 504, 512, 513, 514, 515, 516, 520, 521, 524, 526, 529, 531, 534, 539, 561, 562, 567],
                    'Cluster1': [18, 19, 20, 28, 29, 39, 42, 43, 51, 54, 55, 84, 90, 94, 95, 96, 110, 117, 119, 122, 134, 138, 149, 182, 183, 203, 243, 245, 282, 293, 330, 345, 347, 350, 351, 369, 375, 376, 377, 395, 401, 449, 451, 461, 465, 466, 467, 496, 498, 499, 536, 537, 538, 544, 546, 549, 550, 551, 552, 554, 576, 584],
                    'Cluster2': [67, 68, 74, 106, 130, 140, 186, 236, 237, 254, 259, 261, 310, 319, 338, 339, 365, 398, 407, 410, 412, 413, 414, 434, 435, 436, 437, 439, 480, 481, 482, 483, 484, 485, 487, 489, 490, 492, 493, 494, 501, 505, 519, 525, 540, 541, 545, 568],
                    'Cluster3': [7, 48, 52, 58, 65, 72, 73, 76, 107, 218, 255, 286, 305, 314, 317, 324, 341, 364, 408, 419, 422, 424, 426, 427, 428, 432, 433, 440, 455, 471, 473, 476, 491, 517, 518, 527, 528, 530, 532, 533, 535, 542, 543, 569, 570, 571, 572, 573],
                    'Cluster4': [1, 16, 22, 24, 25, 26, 27, 30, 31, 33, 41, 60, 62, 86, 88, 91, 97, 98, 99, 101, 108, 112, 120, 121, 135, 142, 150, 151, 174, 187, 188, 192, 193, 204, 205, 206, 210, 226, 233, 244, 256, 257, 276, 279, 283, 284, 335, 346, 378, 384, 574, 578, 579, 580, 581],
                    'Cluster5': [5, 115, 123, 126, 214, 222, 223, 249, 252, 253, 263, 264, 265, 268, 269, 275, 277, 278, 292, 294, 295, 296, 301, 308, 309, 311, 318, 326, 327, 328, 329, 343, 359, 362, 371, 374, 385, 563, 565, 566, 582, 583],
                    'Cluster6': [12, 21, 23, 38, 53, 61, 78, 79, 80, 81, 82, 87, 89, 93, 118, 132, 133, 136, 143, 144, 147, 148, 172, 175, 176, 177, 180, 181, 189, 190, 191, 211, 213, 216, 224, 225, 232, 242, 246, 266, 270, 271, 272, 273, 289, 297, 303, 304, 306, 312, 316, 344, 352, 370, 389, 390, 392, 393, 405, 444, 445, 446, 447, 448, 450, 452, 453, 454, 456, 457, 458, 459, 460, 462, 463, 464, 477, 495, 497, 508, 509, 510, 511, 522, 523],
                    'Cluster7': [0, 2, 3, 6, 8, 10, 14, 17, 36, 40, 47, 49, 57, 69, 70, 71, 75, 77, 116, 168, 184, 197, 198, 201, 207, 215, 217, 238, 250, 251, 260, 274, 285, 290, 320, 334, 337, 340, 342, 348, 388, 404, 502, 506, 507, 547, 548, 555, 558, 559, 560, 564],
                    'Cluster8': [9, 15, 50, 56, 59, 83, 85, 114, 125, 128, 137, 139, 165, 170, 185, 200, 202, 239, 280, 281, 321, 323, 332, 349, 360, 361, 366, 386, 387, 397, 399, 470, 503, 553, 556, 557, 575, 577],
                    'Cluster9': [32, 34, 35, 44, 45, 46, 63, 100, 102, 104, 105, 111, 113, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 178, 179, 195, 196, 212, 227, 247, 248, 258, 262, 267, 291, 307, 313, 331, 353, 354, 355, 372, 379, 380, 381, 382, 383, 396, 402],
                    'Cluster10': [92, 103, 145, 146, 152, 158, 171, 173, 194, 228, 229, 230, 231, 234, 235, 288, 298, 299, 300, 302, 322, 394, 400],
                    'Cluster11': [4, 11, 13, 66, 109, 124, 131, 141, 368, 373, 391]
                }
        
    
    groups1 = list(dict_hc.keys())
    groups2 = list(dict_kmeans.keys())
    matrix = np.zeros((len(groups1), len(groups2)), dtype=int)
    # Calculate Rand index
    for i, g1 in enumerate(groups1):
        for j, g2 in enumerate(groups2):
            intersection = set(dict_hc[g1]) & set(dict_kmeans[g2])
            matrix[i][j] = len(intersection)
    df = pd.DataFrame(matrix, index=groups1, columns=groups2)

    total_samples = np.sum(matrix)
    a = 0
    for value in matrix.flatten():
        if value > 1:
            a += (value * (value - 1)) // 2 
    total_pairs = (total_samples * (total_samples - 1)) // 2
    b = total_pairs - np.sum([(np.sum(row) * (np.sum(row) - 1)) // 2 for row in matrix]) \
        - np.sum([(np.sum(col) * (np.sum(col) - 1)) // 2 for col in matrix.T]) + a
    rand_index = (a + b) / total_pairs
    print(f'Rand index: {rand_index}')
    return df

def get_sctter_info_for_origin(df):
    grouped = df.groupby('Cluster')
    new_dfs = []
    for group_name, group_data in grouped:
        group_data = group_data.drop(columns=['Cluster'])
        group_data.columns = [f"{group_name}_{col}" for col in group_data.columns]
        new_dfs.append(group_data.reset_index(drop=True))
    result = pd.concat(new_dfs, axis=1)
    # print(result)
    return result

def get_rank_score(DFT_df):
    df=pd.DataFrame()
    def get_name(name):
        match=re.match(r'^cluster\d+_mol(\d+)_',name)
        if match:
            mol=match.group(1)
            return int(mol)
    df['molecule']=DFT_df['Reactant'].apply(lambda x: get_name(x))
    df['SMILES']=df['molecule'].apply(lambda x: pd.read_csv('./data/onlyNLi.csv')['canonicalsmiles'][x])
    df['precursor_scscore']=df['molecule'].apply(lambda x: get_scscore(pd.read_csv('./data/onlyNH.csv')['canonicalsmiles'][x]))
    df['scscore']=df['SMILES'].apply(lambda x: get_scscore(x))
    df['anode_limit']=DFT_df['AnodeLimit(V)']
    df['element_friendliness']=df['SMILES'].apply(lambda x: get_element_freindless(x,get_all_elements(pd.read_csv('./data/onlyNLi.csv')))[1])
    df['capacity(mAh/g)']=df['SMILES'].apply(lambda x :get_specific_capacity(x))
    df_real=df.copy()
    df[['precursor_scscore','scscore','anode_limit','element_friendliness','capacity(mAh/g)']] = df[['precursor_scscore','scscore','anode_limit','element_friendliness','capacity(mAh/g)']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['total_score']=df['precursor_scscore']+df['scscore']+df['anode_limit']+df['element_friendliness']+df['capacity(mAh/g)']
    df=df.sort_values(by='total_score',ascending=False)
    return df_real,df

if __name__ == '__main__':
    print(get_scscore('CB(O)O'))