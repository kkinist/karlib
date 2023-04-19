#!/usr/bin/env python3
'''
Use Gaussian for robust geometry optimization
command-line argument is a Gaussian input file
'''
import sys, re, os, subprocess
sys.path.insert(0, '/home/irikura/bin')
import chem_subs as chem
import gaussian_subs as gau

try:
    ginput = sys.argv[1]
except IndexError:
    print('Usage: strong_opt.py <name of gaussian gjf file> [max # of attempts]')
    sys.exit(1)
try:
    nmax = int(sys.argv[2])
except IndexError:
    nmax = 3

# 'saddle' should be -1 when no vibrations are computed
# Otherwise, it should be 0 for energy minimum and 1 for TS
# halt cleanly if file "strong_gopt.stop" is found
saddle = 0  # set saddle = order of desired saddle point (1 for TS)

for itry in range(nmax):
    # run the input file
    goutput = ginput.replace('.gjf', '.out')
    with open(goutput, 'w') as OUT:
        INP = open(ginput, 'r')
        subprocess.run('g16', stdin=INP, stdout=OUT, stderr=subprocess.STDOUT)
        INP.close()
        # check for halt file
        if os.path.exists('strong_gopt.stop'):
            print('Halt because file "strong_gopt.stop" detected')
            sys.exit()

    # check for success
    with open(goutput, 'r') as OUT:
        OK = gau.opt_success(OUT)
        if OK:
            # check number of imaginary frequencies
            nimag = gau.get_nimag(OUT)
            if nimag == saddle:
                # successful optimization
                break   # from itry loop

    # determine failure mode and try to fix
    Gjf = gau.GauInput(ginput)
    fmode = gau.failure_mode(goutput)
    # get the lowest-energy coordinates
    xyz = gau.minxyz(goutput)
    if ('empty' in fmode) or (len(xyz) == 0):
        # the input file lacked atoms, or the output file was lame
        # try reading coordinates from the checkpoint file
        Gjf.set_keyword('geom', 'check')
        Gjf.remove_all_atoms()
    if len(xyz):
        # found some coordinates; install them
        Gjf.LGeom.copyxyz(chem.Geometry(xyz, intype='DataFrame'))
    if 'linbend' in fmode:
        # try cartesian coordinates
        Gjf.set_keyword('opt', 'cart')
    if 'endbas' in fmode:
        # maybe the final blank line is missing after @basis 
        Gjf.trailer += '\n'
    # calculate initial force constants
    Gjf.set_keyword('opt', 'calcfc', add=True)
    # modify for transition state
    if saddle == 1:
        Gjf.set_keyword('opt', 'TS', add=True)
    if saddle > 1:
        Gjf.set_keyword('opt', 'saddle={:d}'.format(saddle), add=True)
    
    # temporarily save the old input file
    subprocess.run(['cp', ginput, 'gjf.bak'])
    # write the new input file
    with open(ginput, 'w') as F:
        F.write(Gjf.to_string())

    # save the old output file (temporarily)
    fbak = goutput.replace('.out', '.bak')
    subprocess.run(['mv', goutput, fbak])
if OK:
    if itry > 0:
        st = 'trials'
    else:
        st = 'trial'
    print('Optimization {:s} succeeded in {:d} {:s}'.format(ginput, itry+1, st))
else:
    print('Optimization {:s} failed despite {:d} attempts'.format(ginput, nmax))
