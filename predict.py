import os 
from openbabel import pybel
import tensorflow as tf 
import ResNet
import argparse




def arg_parser():
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file_format','-ftype',required=True,type=str,help='File Format of Protein Structure like: mol2,pdb..etc. All file format supported by Open Babel is supported')
    parser.add_argument('--mode','-m',required=True,type=int,help='Mode 0 is for single protein structure. Mode 1 is for multiple protein structure')
    parser.add_argument('--input_path','-i',required=True,type=str,help='For mode 0 provide absolute or relative path for protein structure. For mode 1 provide absolute or relative path for folder containing protein structure')
    parser.add_argument('--model_path','-mpath',required=True,type=str,help='Provide models absolute or relative path of model')
    parser.add_argument('--output_format','-otype',required=False,type=str,default='mol2',help='Provide the output format for predicted binding side. All formats supported by Open Babel')
    parser.add_argument('--output_path','-o',required=False,type=str,default='output',help='path to model output')
    parser.add_argument('--gpu','-gpu',required=False,type=str,help='Provide GPU device if you want to use GPU like: 0 or 1 or 2 etc.')
    return parser.parse_args()
def main():
    # print(pybel.outformats)

    args=arg_parser()
    if args.mode not in [0,1]:
        raise ValueError('Please Enter Valid value for mode')
    elif args.mode==0:
        if not os.path.isfile(args.input_path):
            raise FileNotFoundError('File Not Found')
    elif args.mode==1:
        if not os.path.isdir(args.input_path):
            raise FileNotFoundError('Folder Not Found')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if args.file_format not in pybel.informats.keys():
        raise ValueError('Enter Valid File Format {}'.format(pybel.informats))
    if args.output_format not in pybel.outformats.keys():
        raise ValueError('Enter Valid Output Format {}'.format(pybel.outformats))
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    model=ResNet.PUResNet()
    model.load_weights('whole_trained_model1.hdf')
    if args.mode==0:
        mol=next(pybel.readfile(args.file_format,args.input_path))
        o_path=os.path.join(args.output_path,os.path.basename(args.input_path))
        if not os.path.exists(o_path):
            os.mkdir(o_path)
        model.save_pocket_mol2(mol,o_path,args.output_format)
    elif args.mode==1:
        for name in os.listdir(args.input_path):
            mol_path=os.path.join(args.input_path,name)
            mol=next(pybel.readfile(args.file_format,mol_path))
            o_path=os.path.join(args.output_path,os.path.basename(args.mol_path))
            if not os.path.exists(o_path):
                os.mkdir(o_path)
            model.save_pocket_mol2(mol,o_path,args.output_format)
if __name__=='__main__':
    main()
