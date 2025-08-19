'''
Credit to https://github.com/CYGNUS-RD/hdf2root/blob/master/hdf2root.py
'''

#!/usr/bin/env python
# Tool to convert H5 Image File in ROOT TH2D
# USAGE:
# convert a list of h5 files:              hdf2root.py Run720/run720*.h5 -o neutrons.root
# convert all the h5 files in a directory: hdf2root.py -d Run720  (will put TH2s into Run720.root)
# -*- coding: utf-8 -*-
import re,sys,os,glob
import numpy as np
from root_numpy import array2hist
import h5py
import ROOT 

def read_image_h5(file):
    with h5py.File(file,'r') as hf:
        data = hf.get('Image')
        np_data = np.array(data)
    return np_data

def read_h5_write_root(fileH5, fileROOT, option='recreate',htype='i'):
    print("file h5 =",fileH5)
    image = read_image_h5(fileH5)
    tf = ROOT.TFile.Open(fileROOT,option)
    (nx,ny) = image.shape
    title = os.path.basename(fileH5).split('.')[0]
    title = title.replace('-','_')
    h2 = ROOT.TH2I(title,title,nx,0,nx,ny,0,ny) if htype=='i' else ROOT.TH2F(title,title,nx,0,nx,ny,0,ny)
    h2.GetXaxis().SetTitle('x')
    h2.GetYaxis().SetTitle('y')
    _ = array2hist(image,h2)
    h2.Write()
    tf.Close()
    return

def h2root_many(h5files,rfname):
    for i,f in enumerate(h5files):
        if not f.endswith('.h5'): 
            continue
        option = 'recreate' if i==0 else 'update'
        print("Saving image ",f," into file ",rfname)
        read_h5_write_root(f,rfname,option)
        
def h2root_dir(h5_dir):
    print("Taking H5 files from ",h5_dir)    
    rfname = os.path.normpath(h5_dir).split('/')[-1]+'.root'
    h5files = []
    for f in os.listdir(h5_dir):
        if f.endswith('.h5'): 
            h5files.append(h5_dir+'/'+f)
    h2root_many(h5files,rfname)
            
if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage='%prog h5file1,...,h5fileN [opts] ')
    parser.add_option('-o', '--outputfile', dest='outputFile', default='image.root', type='string', help='name of the output ROOT file')
    parser.add_option('-d', '--dir', dest='dirWithH5s', default=None, type='string', help='directory where to look for H5 files')
    (options, args) = parser.parse_args()

    if len(args)>0:
        outputFile = options.outputFile
        print("Converting H5 files: ", ", ".join(args)," into ", outputFile)
        h2root_many(args,outputFile)
    else:
        if options.dirWithH5s==None:
            print("If you don't specify a list of files, you need to specify a directory where the files are. Exiting.")
            exit(0)
        else:
            h2root_dir(options.dirWithH5s)