# Routines specific to CFOUR (vers. 2.1)
# Karl Irikura
import sys, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chem_subs as chem

def read_energies(fhandl, etype='target'):
    '''
    Read all energies in file of type 'etype'
    Return a DataFrame with columns:
      (1) line number
      (2) byte number
      (3) number of cycles (if applicable),
      (4) energy energy (hartree)
    'etype' values handled: ['hf', 'ccsd', 'ccsd(t)', 'mp2']
    '''
    fhandl = chem.ensure_file_handle(fhandl)
    byte_start = fhandl.tell()
    fhandl.seek(0)  # rewind file
    energy = []
    ncycle = []
    fline = []
    fpos = []
    lineno = 0
    etype = etype.upper()
    noblock = ['CCSD(T)', 'MP2']  # list of etypes with one-line "blocks" 
    if etype == 'TARGET':
        # determine calclevel
        etype = read_param(fhandl, 'calclevel')
    if etype == 'HF':
        regx_e = re.compile(r'^\s+E\((?:SCF|ROHF)\)=\s+(-\d+\.\d+)\s+[-]?\d+\.\d+D-\d+$')
        regx_iter = re.compile(r'^\s+(\d+)\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+D-\d+$')
        regx_inblock = re.compile('Iteration         Total Energy            Largest Density Difference')
    elif etype == 'CCSD':
        regx_e = re.compile(r'^\s+E\(CCSD\)\s+=\s+(-\d+\.\d+)$')
        regx_iter = re.compile(r'^\s+(\d+)\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+')
        regx_inblock = re.compile('Summary of iterative solution of CC equations')
    elif etype == 'CCSD(T)':
        regx_e = re.compile(r'^\s+E\(CCSD\(T\)\)\s+=\s+(-\d+\.\d+)$')
    elif etype == 'MP2':
        regx_e = re.compile(r'^\s+Total MP2 energy\s+=\s+(-\d+\.\d+)')
    else:
        chem.print_err('', f'unrecognized energy type = {etype}')
    if etype in noblock:
        regx_iter = re.compile('Method is not iterated')
        regx_inblock = re.compile('Block is only one line')

    niter = 0
    inblock = etype in noblock
    while True:
        line = fhandl.readline()
        if not line:
            break
        lineno += 1
        if inblock:
            m = regx_iter.match(line)
            if m:
                # iteration count (starts at zero)
                niter = int(m.group(1))
            m = regx_e.search(line)
            if m:
                # found an energy
                energy.append(float(m.group(1)))
                ncycle.append(niter)
                fline.append(lineno)
                fpos.append(fhandl.tell())
                # reset things
                niter = 0
                if etype not in noblock:
                    inblock = False
        if regx_inblock.search(line):
            inblock = True
    if etype in noblock:
        data = list(zip(fline, fpos, energy))
        cols = ['line', 'byte', etype]
    else:
        data = list(zip(fline, fpos, ncycle, energy))
        cols = ['line', 'byte', 'Cycles', etype]
    df = pd.DataFrame(data=data, columns=cols)
    fhandl.seek(byte_start) # restore file pointer to original position
    return df
##
def final_energy(fhandl, etype='target'):
    # return the last energy found in the file
    # 'etype' can be any of ['HF', 'CCSD', 'CCSD(T)', 'target'],
    # where 'target' means the energy requested on the command line
    dfe = read_energies(fhandl, etype)
    etype = etype.upper()
    elast = dfe[dfe.columns[-1]].iloc[-1]  # take energy from last column
    return elast
##
def spin_contam(fhandl, etype):
    # return a DataFrame of spin contamination values = <S**2> - S(S+1)
    fhandl = chem.ensure_file_handle(fhandl)
    ok = ['hf', 'mp2', 'ccsd']  # list of methods coded
    if etype.lower() not in ok:
        return None
    byte_start = fhandl.tell()
    fhandl.seek(0)  # rewind file
    re_mult = re.compile('       MULTIPLICTY          IMULTP \s+(\d+)')
    re_proj = re.compile('^             Projected <0\|S\^2 exp\(T\)\|0> =\s+(\d+\.\d+)\.$')
    if etype == 'hf':
        re_start = re.compile('SCF has converged')
        re_proj = re.compile('\s+The expectation value of S\*\*2 is\s+([-]?\d+\.\d+)')
    elif etype == 'mp2':
        re_start = re.compile('Total MP2 energy')
    elif etype == 'ccsd':
        re_start = re.compile('CCSD.*energy will be calculated')
    lineno = []
    contam = []
    in_block = False
    lno = 0  # line number
    smult = None  # spin multiplicity = 2*S + 1
    starg = None  # value of S*(S+1)
    for line in fhandl:
        if smult is None:
            m = re_mult.match(line)
            if m:
                smult = float(m.group(1))
                starg = (smult * smult - 1) / 4
        if in_block:
            m = re_proj.match(line)
            if m:
                lineno.append(lno)
                contam.append(float(m.group(1)) - starg)
                in_block = False
        if re_start.search(line):
            in_block = True
        lno += 1
    df = pd.DataFrame({'line': lineno, 'contam': contam})
    fhandl.seek(byte_start) # restore file pointer to original position
    return df
##
def final_spin_contam(fhandl, etype):
    # return the last spin contamination found in the file
    dfc = spin_contam(fhandl, etype)
    if dfc is None:
        return None
    dfc = dfc.sort_values('line')
    clast = dfc.contam.values[-1]
    return clast
##
def read_control_parameters(fhandl):
    # Read the table of "CFOUR Control Parameters" and return a DataFrame
    # Also return the job title
    fhandl = chem.ensure_file_handle(fhandl)
    re_start = re.compile('^\s+CFOUR Control Parameters\s+$')
    re_end = re.compile('Job Title : (.+)')
    re_capalpha = re.compile('[A-Z]')
    re_trail = re.compile('[*]+$')
    xname = []  # 'External Nazme'
    iname = []  # 'Internal Name'
    value = []  # 'Value' (up to left bracket, if any)
    inblock = False
    byte_start = fhandl.tell()
    fhandl.seek(0)  # rewind file
    for line in fhandl:
        if inblock:
            words = line.split()
            # valid external name is all caps
            if (words[0] == words[0].upper()) and (re_capalpha.search(words[0])):
                if words[0] == 'B':
                    # special case: beta orbital occupation
                    value[-1] = '/'.join([value[-1], line.strip()])
                else:
                    xname.append(words[0])
                    iname.append(words[1])
                    v = ''.join(words[2:]).strip()
                    # remove left bracket and anything following
                    i = v.find('[')
                    if i > -1:
                        v = v[:i]
                    value.append(v)
                # remove any trailing asterisks
                m = re_trail.search(value[-1])
                if m:
                    value[-1] = value[-1].replace(m.group(0), '')
            m = re_end.search(line)
            if m:
                inblock = False
                title = m.group(1)
        m = re_start.match(line)
        if m:
            inblock = True
    df = pd.DataFrame({'param': xname, 'code': iname, 'value': value})
    fhandl.seek(byte_start) # restore file pointer to original position
    return df, title
##
def read_param(file, param):
    # Return the value listed for a 'CFOUR Control Parameter'
    df, title = read_control_parameters(file)
    param = param.upper()
    value = df[df.param == param]['value'].iloc[0]
    return value
##
def read_harmonic_freqs(file, vib_only=True):
    # return a DataFrame; last set of frequencies in file
    # frequencies may be imaginary; convert here to negative float
    # setting 'vib_only' = False includes the rotations and translations
    rx_start = re.compile('^\s+Normal Coordinate Analysis\s*$')
    rx_end = re.compile('^\s+Normal Coordinates\s*$')
    rx_freq = re.compile('^\s+(\S+)\s+(\d+\.\d+)([ i])\s+(\d+\.\d+)\s+([A-Z]+)')
    inblock = False
    irrep = []
    freq = []  # cm-1
    inten = []  # km/mol
    vtype = []  # VIBRATION, ROTATION, or TRANSLATION
    with open(file, 'r') as F:
        for line in F:
            if inblock:
                if rx_end.match(line):
                    inblock = False
                    continue
                m = rx_freq.match(line)
                if m:
                    words = line.split()  # four fields
                    irrep.append(words[0].replace('-', ''))  # replace '----' with ''
                    if m.group(3) == 'i':
                        freq.append(-1 * float(m.group(2)))
                    else:
                        freq.append(float(words[1]))
                    inten.append(float(words[2]))
                    vtype.append(words[3])
            if rx_start.match(line):
                inblock = True
    df = pd.DataFrame({'Freq': freq, 'IR inten': inten, 'Irrep': irrep, 'Type': vtype})
    if vib_only:
        # exclude rotations and translations
        df = df[df.Type == 'VIBRATION'].drop(columns=['Type'])
    return df
##
############ stuff below from c4_subs.py on gamba #############
# Routines for use in parsing CFOUR output files.
# Karl Irikura, 11/7/12
# last change 11/7/12
#
def read_natoms( fhandl ):
    # read the number of atoms
    fhandl.seek( 0 )    # rewind file
    natom = 0
    indist = False
    regx = re.compile( 'Interatomic distance matrix' )
    endr = re.compile( 'Rotational constants' )
    for line in fhandl:
        mch1 = regx.search( line )
        indist = indist or mch1
        if indist:
            mch2 = endr.search( line )
            if mch2:
                break
            #print line
            nums = re.findall( r'\[\s*(\d+)\]', line )
            nums = map( int, nums )
            natom = max( natom, nums )
    return natom[0]
##
def read_omega( fhandl ):
    # read harmonic frequencies from a Gaussian09 output file
    # not for VPT2 outputs; use read_fundamental instead
    # return a list of the values (8/27/2012)
    fhandl.seek( 0 )    # rewind file
    omega = []
    regx = re.compile( ' Frequencies --' )
    for line in fhandl:
        mch = regx.search( line )
        if mch:
            line = line.split()
            omega.extend( line[2:] )
    return map( float, omega )
##
def read_zpe( fhandl ):
    # read ZPE from harmonic or VPT2 output file
    fhandl.seek( 0 )
    zpe = 0.0
    regx_har = re.compile( 'Zero-point vibrational energy:' )    # in kJ/mol
    regx_anh = re.compile( 'Harm\+VPT2             :' )    # in cm-1
    for line in fhandl:
        mch = regx_har.search( line )
        if mch:
            line = line.split()
            zpe = float( line[-2] )
            zpe *= au_wavenumber / au_kjmol    # convert to cm-1
            continue
        mch = regx_anh.search( line )
        if mch:
            line = line.split()
            zpe = float( line[-1] )    # already in cm-1
            continue
    return zpe
##
def read_fundamental( fhandl ):
    # read harmonic and anharmonic frequencies
    fhandl.seek( 0 )    # rewind file
    omega = []
    nu = []
    regx_begin = re.compile( 'HARMONIC AND FUNDAMENTAL FREQUENCIES' )
    regx_end = re.compile( 'ZERO-POINT VIBRATIONAL ENERGIES' )
    regx_float = re.compile( '\d\.\d' )
    in_fund = False
    for line in fhandl:
        mch = regx_end.search( line )
        if mch:
            in_fund = False
        if in_fund:
            mch = regx_float.search( line )
            if mch:
                # parse a line
                line = line.split()
                omega.append( float( line[1] ) )
                nu.append( float( line[2] ) )
            continue
        mch = regx_begin.search( line )
        if mch:
            in_fund = True
    return (omega, nu)
##
def read_overtones( fhandl ):
    # read 2-0 overtones and 1+1' combination bands; return them as lists of float
    # they are 2D lists
    # by default, CFOUR prints levels up to three quanta
    fhandl.seek( 0 )
    overt = []
    combi = []
    regx_ov = re.compile( 'MODE MODE MODE MODE MODE' )
    regx_end = re.compile( 'Dipole moment function written' )
    regx_float = re.compile( '\d\.\d' )
    in_ov = False
    for line in fhandl:
        mch = regx_end.search( line )
        if mch:
            break
        if in_ov:
            mch = regx_float.search( line )
            if not mch:
                continue
            # remove any (problematic) asterisks
            line = line.replace( '*', ' ' )
            line = line.split()
            quanta = sum( map( int, line[5:10] ) )
            if quanta != 2:
                continue
            if int( line[5] ) == 2:
                # read one (2-0) overtone
                a = [ int(line[0])-6, float(line[10]) ]    # mode number, then anharmonic value
                overt.append( a )
            else:
                # read a 1+1' combination
                a = [ int(line[0])-6, int(line[1])-6, float(line[10]) ]    # mode numbers, then anharmonic value
                combi.append( a )
            continue
        mch = regx_ov.search( line )
        if mch:
            # overtones/combinations are combined in CFOUR output
            in_ov = True
            continue
    return ( overt, combi )
##        
def read_xij( fhandl ):
    # read vibrational anharmonicity constants
    # return Xij as 2D list in lower-triangular form
    # CFOUR does not report X0; return -9999 to indicate that
    # it must be computed from ZPE and Xij's
    fhandl.seek( 0 )
    x = []
    x0 = -9999
    regx_begin = re.compile( 'ANHARMONICITY CONSTANTS X\(ij\)' )
    regx_end = re.compile( 'HARMONIC AND FUNDAMENTAL FREQUENCIES' )
    regx_float = re.compile( '\d\.\d' )
    in_x = False
    for line in fhandl:
        mch = regx_end.search( line )
        if mch:
            in_x = False
            continue
        if in_x:
            mch = regx_float.search( line )
            if mch:
                # parse this line
                line = line.replace( '*', ' ' ) # remove any asterisk
                line = line.split()
                i = int( line[0] ) - 6    # 1-adjusted row number
                if i > len( x ):
                    x.append( [] )    # add a new row with leading zeros
                    for j in range( i-1 ):
                                                x[i-1].append( 0 )
                x[i-1].append( line[2] )    # add to new or existing row
                continue
        mch = regx_begin.search( line )
        if mch:
            in_x = True
            continue
    # convert from string to float
    n = len( x )
    for i in range( n ):
        x[i] = map( float, x[i] )
    # fill empty elements with zero
    for i in range( n ):
        for j in range( len( x[i] ), n ):
            x[i].append( 0.0 )
    # now transpose to make lower-triangular
    x = transpose( x )
    return (x, x0)
##
