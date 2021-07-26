import openbabel
import pybel
import numpy as np

def get_fingerprint(file_type,file_path):
    '''file type: file format that is accepted by Openbabel
        file_path: path to file
        returns fingerprint based on MACCS'''
    mol=next(pybel.readfile(file_type,file_path))
    fingerprint=[]
    residues=[x for x in openbabel.OBResidueIter(mol.OBMol)]
    for i in range(len(residues)-3):
        mers=residues[i:i+3]
        mol1=openbabel.OBMol()
        for r in mers:
            for atoms in openbabel.OBResidueAtomIter(r):
                mol1.AddAtom(atoms)
        mol1.ConnectTheDots()
        mol2=pybel.Molecule(mol1)
        fp=[1 if i in list(mol2.calcfp('MACCS').bits) else 0 for i in range(167)]
        fingerprint.append(fp)
    return np.asarray(fingerprint)

def calculate_tanimato(fp1,fp2):
    '''Calculated tanimato cofficient between two fingerprint fp1 and fp2'''
    intersection=np.sum(fp1*fp2)
    denominator=np.sum(fp1)+np.sum(fp2)-intersection
    return intersection/denominator

def get_tanimato_cofficient(file_type1,file_path1,file_type2,file_path2):
    '''file_type 1 and 2 : file format accepted by openbabel
        file_path 1 and 2 : path to file 1 and 2 respectively
        returns tanimato cofficient '''
    fp1=get_fingerprint(file_type1,file_path1)
    fp2=get_fingerprint(file_type2,file_path2)
    l_fp1,l_fp2=len(fp1),len(fp2)
    if l_fp1>l_fp2:
        temp=[]
        for i in range(l_fp1-l_fp2):
            tscore=calculate_tanimato(fp1[i:i+l_fp2].flatten(),fp2.flatten())
            temp.append(tscore)
        return np.max(temp)
    elif l_fp1<l_fp2:
        temp=[]
        for i in range(l_fp2-l_fp1):
            tscore=calculate_tanimato(fp2[i:i+l_fp1].flatten(),fp1.flatten())
            temp.append(tscore)
        return np.max(temp)
    else:
        score=calculate_tanimato(fp2.flatten(),fp1.flatten())
        return score  