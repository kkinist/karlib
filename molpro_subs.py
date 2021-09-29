# routines for use with MOLPRO 2012 files
# 5/28/2020 KKI start
#
import re, sys
import pandas as pd
import numpy as np
#sys.path.insert(1, r'C:\Users\irikura\Documents\GitHub\bin3dev')
import chem_subs as chem
##
SPINMULT = {0: 'Singlet', 1: 'Triplet', 0.5: 'Doublet', 1.5: 'Quartet', 2: 'Quintet'}
MULTSPIN = {v: k for k, v in SPINMULT.items()}
LAMBDA = {0: 'Sigma', 1: 'Pi   ', 2: 'Delta', 3: 'Phi  ', 4: 'Gamma'}
LSYMB = ['S', 'P', 'D', 'F', 'G']
OMEGA = '\N{GREEK CAPITAL LETTER OMEGA}'
LAMDA = '\N{GREEK CAPITAL LETTER LAMDA}'
SIGMA = '\N{GREEK CAPITAL LETTER SIGMA}'
PI = '\N{GREEK CAPITAL LETTER PI}'
DELTA = '\N{GREEK CAPITAL LETTER DELTA}'
PHI = '\N{GREEK CAPITAL LETTER PHI}'
GAMMA = '\N{GREEK CAPITAL LETTER GAMMA}'
GLAMBDA = {0: SIGMA, 1: PI, 2: DELTA, 3: PHI, 4: GAMMA}
##
class MULTI:
    # A list of lines of MCSCF output from "MULTI"
    # also some results of parsing those lines
    # 'PG' is name of computational point group (optional)
    def __init__(self, linebuf, PG=None):
        self.lines = linebuf
        self.PG = PG
        self.nfrozen = self.nfrozen()
        self.norb = self.nactorb()
        self.groups = self.parseGroups()
        self.results = self.parseResults()
        self.termLabels()
        self.NOs = self.natorb_info()
        self.civec = self.parseMULTIcivec()
    def print(self):
        print('{:d} closed-shell orbitals'.format(self.nfrozen))
        print('{:d} active orbitals'.format(self.norb))
        print('Groups:')
        for g in self.groups:
            print(g)
    def printlines(self):
        # print the lines from MOLPRO output
        print('\n'.join(self.lines))
        return
    def nfrozen(self):
        # the number of frozen-core aka closed-shell orbitals
        # If it's zero, that is not stated in the MOLPRO output
        rx = re.compile(r' Number of closed-shell orbitals:\s+(\d+) ')
        for line in self.lines:
            m = rx.match(line)
            if m:
                return int(m.group(1))
        return 0
    def nactorb(self, irreps=False):
        # the number of active orbitals
        # if irreps==True, instead of an integer return the irrep list
        rx = re.compile(r' Number of active  orbitals:\s+(\d+) ')
        rx_irr = re.compile(r'\((\s+\d)+\s*\)')
        for line in self.lines:
            m = rx.match(line)
            if m:
                if irreps:
                    m = rx_irr.search(line)
                    words = m.group(0).split()
                    return [int(n) for n in words[1:-1]]
                else:
                    return int(m.group(1))
        return None
    def parseGroups(self):
        # A "group" here is labeled "State symmetry" in the output,
        # because that name is too confusing without context.
        # Return a list of dicts with basic parameters
        glist = []
        rx_g = re.compile(r' State symmetry (\d+)\s*$')
        rx_end = re.compile(r' orbitals read from record| Orbital guess generated')
        rx_nelec = re.compile(r' Number of (?:active )?electrons:\s+(\d+)\s+Spin symmetry=(\w+)\s+Space symmetry=(\d)')
        rx_nstates = re.compile(r' Number of states:\s+(\d+)')
        ingroup = False
        parms = {}
        for line in self.lines:
            if ingroup:
                if rx_end.search(line):
                    # all done
                    glist.append(parms.copy())
                    break
                m = rx_nelec.match(line)
                if m:
                    parms['nElec'] = int(m.group(1))
                    parms['spinLabel'] = m.group(2)
                    parms['irrep'] = int(m.group(3))
                m = rx_nstates.match(line)
                if m:
                    parms['nStates'] = int(m.group(1))
            m = rx_g.match(line)
            if m:
                ingroup = True
                gnum = int(m.group(1))  # the "State symmetry" number
                if gnum > 1:
                    # remember the previous group
                    glist.append(parms)
                # start new group
                parms = {'gNum': gnum}
        return glist
    def parseResults(self):
        # Return a DataFrame with state information
        gnum = []
        size = []
        spin = []
        irrep = []
        for g in self.groups:
            gnum.extend([g['gNum']] * g['nStates'])
            spin.extend([g['spinLabel']] * g['nStates'])
            irrep.extend([g['irrep']] * g['nStates'])
            size.extend(['{:d}/{:d}'.format(g['nElec'], self.norb)] * g['nStates'])
        # get state labels and energies
        rx_e = re.compile(r' !MCSCF STATE\s*(\d+\.\d) (?:\S+ )?Energy\s+([-]?\d+\.\d+)')
        rx_dip = re.compile(r' !MCSCF STATE\s*(\d+\.\d) (?:\S+ )?Dipole moment\s+')
        lbl = []
        energy = []
        dipx = []
        dipy = []
        dipz = []
        # some or all of the angular momentum expectation values (below) may be absent
        rx_expec = re.compile(r' !MCSCF expec\s+<(\d+\.\d)(?: \S+)?\|(L...)?\|(\d+\.\d)(?: \S+)?>\s+([-]?\d+\.\d+)')
        lexpec = {'LXLX': [], 'LYLY': [], 'LZLZ': [], 'L**2': []}
        for line in self.lines:
            m = rx_e.match(line)
            if m:
                # state energy
                lbl.append(m.group(1))
                energy.append(float(m.group(2)))
            m = rx_dip.match(line)
            if m:
                # dipole moment in a.u.
                if m.group(1) != lbl[-1]:
                    # state label is different than it was for preceding energy!
                    print('*** Error: Dipole label is {:s} but should be {:s}'.format(m.group(1), lbl[-1]))
                    sys.exit(0)
                words = line.split()
                dipx.append(float(words[-3]))
                dipy.append(float(words[-2]))
                dipz.append(float(words[-1]))
            m = rx_expec.match(line)
            if m:
                if m.group(1) != m.group(3):
                    # ignore off-diagonal values
                    continue
                n = len(lexpec[m.group(2)])
                if lbl[n] != m.group(1):
                    print('*** Error: expected label {:s} for this line:'.format(lbl[n]))
                    print(line)
                    sys.exit(0)
                lexpec[m.group(2)].append(float(m.group(4)))
        df = pd.DataFrame({'Group': gnum, 'Size': size, 'Spin': spin, 'Irrep': irrep,
                          'Label': lbl, 'Energy': energy, 'dipX': dipx, 'dipY': dipy,
                          'dipZ': dipz})
        df['Dipole'] = np.sqrt(df.dipX ** 2 + df.dipY ** 2 + df.dipZ ** 2)
        if lexpec['LXLX']:
            df['LxLx'] = lexpec['LXLX']
        if lexpec['LYLY']:
            df['LyLy'] = lexpec['LYLY']
        if lexpec['LZLZ']:
            df['LzLz'] = lexpec['LZLZ']
        if lexpec['L**2']:
            df['L**2'] = lexpec['L**2']
        return df
    def termLabels(self, greek=True, hyphen=False, quiet=True):
        try:
            self.results = termLabels(self.results, greek=greek, hyphen=hyphen, PG=self.PG)
        except:
            if not quiet:
                chem.print_err('Unable to assign term symbol', halt=False)
        return
    def natorb_info(self):
        # return a DataFrame describing the natural orbitals
        rx_NO = re.compile('\s+NATURAL ORBITALS')
        rx_end = re.compile('Total charge:')
        rx_data = re.compile('\s+\d+\.\d\s+[-]?\d\.\d+')
        cols = ['Orb', 'Occ', 'Active', 'E', 'Compos', 'Terse']
        orb = []  # orbital label, e.g. '3.1'
        occ = []  # occupation 
        e = []    # energy
        comp = []  # composition as list of tuples: (center, type#, type, coeff)
        terse = [] # concise description
        irrep = []  # irrep number
        inNO = False
        for line in self.lines:
            if inNO:
                if rx_data.match(line):
                    # extract the data
                    words = line.split()
                    orb.append(words.pop(0))
                    irrep.append(int(orb[-1].split('.')[-1]))
                    occ.append(float(words.pop(0)))
                    e.append(float(words.pop(0)))
                    # compositions remain
                    c = []
                    while words:
                        if len(words) % 4:
                            # number of fields does not make sense
                            chem.printerr('', 'Composition fields not a multiple of 4: {:d}'.format(len(words)))
                        c.append((int(words[0]), int(words[1]), words[2], float(words[3])))
                        words = words[4:]
                    comp.append(c)
                    # create concise description of this orbital
                    ac = np.array([abs(x[3]) for x in c])
                    cmax = ac.max()
                    idx = np.flip(np.argsort(ac))
                    descr = ''
                    s = ''
                    for i in idx:
                        x = c[i]
                        if abs(x[3]) > cmax * 2/3:
                            # include in the description
                            if len(descr):
                                # include the sign relative to the leading component
                                if x[3] * c[idx[0]][3] > 0:
                                    s = '+'
                                else:
                                    s = '-'
                            descr += '{:s}{:d}.{:d}.{:s}'.format(s, x[0], x[1], x[2])
                    terse.append(descr)
                if rx_end.search(line):
                    inNO = False
            if rx_NO.match(line):
                inNO = True
        # mark active orbitals starting from the end
        vocc = self.nactorb(irreps=True)
        activ = [False] * len(orb)
        for i in reversed(range(len(irrep))):
            j = irrep[i] - 1
            if vocc[j] > 0:
                activ[i] = True
                vocc[j] -= 1
                if occ[i] == 2.00000:
                    chem.print_err('', f'NO #{orb[i]} is active and doubly occupied', halt=False)
        if np.array(vocc).any():
            vocc = self.nactorb(irreps=True)
            nirrep = np.unique(irrep, return_counts=True)[1]  # np.unique returns two arrays
            chem.print_err('', f'Number of active orbitals {vocc} exceeds number of NOs {nirrep}')
        data = {k: v for k, v in zip(cols, [orb, occ, activ, e, comp, terse])}
        df = pd.DataFrame(data)
        if len(df):
            return df
        else:
            return None
    def parseMULTIcivec(self, thresh=0.1):
        # For each MCSCF state, assign a list of configurations and
        #   corresponding list of coefficients (must exceed 'thresh')
        rx_hdr = re.compile(r'CI (?:vector for state|Coefficients of) symmetry\s*(\d+)')
        rx_end = re.compile('TOTAL ENERGIES|Energy: ')
        rx_data = re.compile('(\s+[20ab]+)+(\s+[-]?\d\.\d+)+')
        indata = False
        civec = []  # list of DataFrames; index specifies state
        # new 2021 format does not list group numbers--assume they are in order
        grp = 0
        for line in self.lines:
            if indata:
                m = rx_data.match(line)
                if m:
                    # parse a line of data
                    words = line.split()
                    # occupation numbers are lumped together by irrep
                    # remove the spaces to make it as in MRCI
                    v = ''.join(words[:-nstate])
                    c = [float(w) for w in words[-nstate:]]
                    for i in range(nstate):
                        if abs(c[i]) > thresh:
                            coeffs[i].append(c[i])
                            vecs[i].append(v)
                if rx_end.search(line):
                    # save the data
                    for i in range(nstate):
                        df = pd.DataFrame({'coeff': coeffs[i], 'occup': vecs[i]})
                        self.comment_occup(df)
                        civec.append(df)
                    indata = False
            m = rx_hdr.search(line)
            if m:
                grp += 1
                group = self.groups[grp - 1]
                #if grp != group['gNum']:
                #    chem.print_err('', 'Found group {:d} but expected {:d}'.format(grp, group['gNum']))
                nstate = group['nStates']
                vecs = [[] for n in range(nstate)] 
                coeffs = [[] for n in range(nstate)] 
                indata = True
        return civec
    def comment_occup(self, df):
        # generate comments describing occupation vectors
        # add a column 'comment' to the DataFrame argument
        if 'comment' in df:
            # there is already a comment; do nothing
            return
        comment = []
        # occupation vector only refers to active orbitals
        dfNO = self.NOs[self.NOs.Active]
        patts = df.occup.values  # occupation strings
        imax = np.argmax(np.abs(df.coeff.values))
        domin = df.iloc[imax].occup  # 'patt' for the dominant configuration
        for patt in patts:
            comm = []
            if patt != domin:
                # non-dominant configuration; compare spins with dominant
                ab = sum([1 for (a,b) in zip(domin,patt) if (a != b) and (b in 'ab')])
            else:
                ab =  ('a' in domin) or ('b' in domin)
            if (patt == domin) or ab:
                if not ab:
                    # dominant is closed shell; just note that it's dominant
                        comm.append('dominant')
                else:
                    # look for unpaired electrons
                    for i, ch in enumerate(patt):
                        if ch == 'a':
                            comm.append('alpha({:s})'.format(dfNO.iloc[i].Terse))
                        if ch == 'b':
                            comm.append('beta({:s})'.format(dfNO.iloc[i].Terse))
            else:
                # comment on pair changes from leading configuration
                lost = []   # '2' became '0'
                gained = [] # '0' became '2'
                for i, (a, b) in enumerate(zip(domin, patt)):
                    if a != b:
                        if a == '2':
                            lost.append(i)
                        else:
                            gained.append(i)
                if len(lost):
                    c = 'lose('
                    for i in lost:
                        c += dfNO.iloc[i].Terse
                    c += ')'
                    comm.append(c)
                if len(gained):
                    c = 'gain('
                    for i in gained:
                        c += dfNO.iloc[i].Terse
                    c += ')'
                    comm.append(c)
            comment.append(' '.join(comm))
        df['comment'] = comment
        return
##
class MRCI:
    # A list of lines of MRCI output 
    # also some results of parsing those lines
    def __init__(self, linebuf):
        self.lines = linebuf
        x = self.basics()
        self.irrep = x[0]
        self.spinLabel = x[1]
        self.ncore = x[2]
        #self.nact = x[3]
        self.results = self.properties()
    def printlines(self):
        print('\n'.join(self.lines))
    def basics(self):
        # get irrep, spin and electron counts
        rx_sym = re.compile(r' Reference symmetry:\s+(\d)\s+(\w+)')
        rx_ncore = re.compile(r' Number of core orbitals:\s+(\d+)')
        #rx_nact = re.compile(r' Number of active\s+orbitals:\s+(\d+)')
        ncore = 0  # may not be stated when ncore == 0
        for line in self.lines:
            m = rx_sym.match(line)
            if m:
                irrep = int(m.group(1))
                spinLabel = m.group(2)
            m = rx_ncore.match(line)
            if m:
                ncore = int(m.group(1))
            #m = rx_nact.match(line)
            #if m:
            #    nact = int(m.group(1))
        #return irrep, spinLabel, ncore, nact
        return irrep, spinLabel, ncore
    def properties(self):
        # return a DataFrame of interesting properties
        rx_energy = re.compile(r' !MRCI STATE\s*(\d+\.\d) Energy\s+([-]?\d+\.\d+)')
        rx_dip = re.compile(r' !MRCI STATE\s*(\d+\.\d) Dipole moment')
        # if there is only one state, "rotated" (below) does not occur
        # "rotated" is after "relaxed" in the output file
        rx_dav = re.compile(r' Cluster corrected energies\s+([-]?\d+\.\d+) \(Davidson, (relaxed|rotated) reference\)')
        energy = []
        edav = []
        lbl = []
        dipx = []
        dipy = []
        dipz = []
        rx_ref = re.compile(r' RESULTS FOR STATE\s*(\d+\.\d)')
        rx_ovl = re.compile(r' Maximum overlap with reference state\s+\d+')
        rx_c0 = re.compile(r' Coefficient of reference function:   C\(0\) =')
        rx_eref = re.compile(r' Reference energy\s+([-]?\d+\.\d+)')
        ref = []  # label for CASSCF state with largest overlap
        c0 = []   # coefficient of ref state
        eref = [] # energy of CASSCF reference state
        reflbl = None
        for line in self.lines:
            m = rx_energy.match(line)
            if m:
                # state energy
                energy.append(float(m.group(2)))
                lbl.append(m.group(1))
            m = rx_dip.match(line)
            if m:
                # dipole moment in a.u.
                if m.group(1) != lbl[-1]:
                    # state label is different than it was for preceding energy!
                    print('*** Error: Dipole label is {:s} but should be {:s}'.format(m.group(1), lbl[-1]))
                    sys.exit(0)
                words = line.split()
                dipx.append(float(words[-3]))
                dipy.append(float(words[-2]))
                dipz.append(float(words[-1]))
            m = rx_dav.match(line)
            if m:
                # want the "energd4" option
                n = len(energy)
                dave = float(m.group(1))
                try:
                    # overwrite earlier value
                    edav[n-1] = dave
                except:
                    # not yet created
                    edav.append(dave)
            m = rx_ref.match(line)
            if m:
                reflbl = m.group(1)
            if rx_ovl.match(line):
                words = line.split()
                # replace the first number in the label with this one
                # (keep the second number, which specifies the irrep)
                reflbl = re.sub('\d+\.', words[-1] + '.', reflbl)
            if rx_c0.match(line):
                words = line.split()
                c0.append(float(words[-2]))  # choose the "rotated" value
                ref.append(reflbl)
            m = rx_eref.match(line)
            if m:
                eref.append(float(m.group(1)))
        # create the DataFrame
        df = pd.DataFrame({'Spin': self.spinLabel, 'Irrep': self.irrep, 
                           'Label': lbl, 'Energy': energy, 'Edav': edav, 
                           'Ncore': self.ncore, 'dipX': dipx, 'dipY': dipy, 
                           'dipZ': dipz, 'Eref': eref})
        df['Dipole'] = np.sqrt(df.dipX**2 + df.dipY**2 + df.dipZ**2)
        df['Ref'] = ref
        df['C0'] = c0
        return df
    def transfer_lz(self, dfcas, etol=1.e-6):
        # assign Lz values to MRCI states from Lz values of dominant MCSCF states
        # dfcas is from MULTI().results
        # 'etol' is the energy-matching requirement for the MCSCF energy and the MRCI "reference" energy
        # No return value: self.results is modified
        spin = self.results.loc[0, 'Spin']
        irrep = self.results.loc[0, 'Irrep']
        subdf = dfcas[(dfcas.Spin == spin) & (dfcas.Irrep == irrep)].copy()
        lzz = []
        term = []
        r = []
        hasR = 'R' in dfcas  # whether bond length is present
        for i, row in self.results.iterrows():
            crow = subdf[(subdf.Label == row.Ref)]
            if hasR:
                r.append(crow.R.values[0])
            if abs(row.Eref - crow.Energy.values[0]) < etol:
                lzz.append(crow.LzLz.values[0])
                term.append(crow.Term.values[0])
            else:
                print('*** wrong reference energy (diff = {:e})'.format(row.Eref - crow.Energy.values[0]))
                print(row.to_frame().T)
                print(crow)
                lzz.append(np.nan)
                term.append(np.nan)
        self.results['LzLz'] = lzz
        self.results['Term'] = term
        if hasR:
            self.results['R'] = r
        return    
    def print(self, reset=False):
        # print the DataFrame of results (may have been externally modified)
        if reset:
            # replace the existing self.results
            self.results = self.properties()
        print(self.results)
        return
##
class SOenergy:
    # A list of the energy-related lines of SO-CI output 
    # also some results of parsing those lines
    def __init__(self, linebuf, E0):
        self.lines = linebuf
        self.E0 = E0
        self.energies = self.get_energies()
        self.results = self.energies  # a synonym to be more consistent with other classes
    def get_energies(self):
        # return a DataFrame of state energies
        re_data = re.compile(r'\s+\d+\s+\d(\s+[-]?\d+\.\d+){6}')
        re_alt = re.compile(r'\s+\d+(\s+[-]?\d+\.\d+){6}')
        Nr = []
        Sym = []
        E = []
        cm = []
        excit = []
        for line in self.lines:
            if re_data.match(line):
                words = line.split()
                Nr.append(int(words[0]))
                Sym.append(int(words[1]))
                E.append(float(words[2]))
                cm.append(float(words[4]))  # E-E0
                excit.append(float(words[6]))  # E-E(1) in cm-1
            if re_alt.match(line):
                words = line.split()
                Nr.append(int(words[0]))
                Sym.append(0)  # not present in half-integer output
                E.append(float(words[1]))
                cm.append(float(words[3]))
                excit.append(float(words[5]))
        if Sym[0] != 0:
            df = pd.DataFrame({'Nr': Nr, 'Irrep': Sym, 'E': E, 'Eshift': cm,
                               'Erel': excit})
        else:
            # omit the 'Irrep' column
            df = pd.DataFrame({'Nr': Nr, 'E': E, 'Eshift': cm, 'Erel': excit})
        # sort by energy
        return df.sort_values(by='E').reset_index(drop=True)
    def printlines(self):
        print('\n'.join(self.lines))
        return
    def recalc_wavenumbers(self):
        # Recompute 'Eshift' and 'Erel' to get precision better than
        # the MOLPRO-output 0.01 cm-1
        self.energies['Eshift'] = chem.AU2CM * (self.energies.E - self.E0)
        self.energies['Erel'] = self.energies.Eshift - self.energies.Eshift.min()
        return
    def collect_degenerate(self, cmtol=1.):
        # return a DataFrame with degenerate levels averaged and counted
        # levels are "degenerate" when their differences <= 'cmtol', in cm-1 units
        #    "degenerate" is transitive, so differences may exceed the tolerance
        # Ignore column "Irrep"
        # Re-calculate wavenumber quantities to increase their precision
        self.recalc_wavenumbers()
        return collect_degenerate(self.energies, cmtol=cmtol)
##
class SOcompos:
    # A list of composition-related lines of SO-CI output
    # also some results of parsing those lines
    def __init__(self, linebuf):
        self.lines = linebuf
        self.basis = self.get_basis()
        self.pct = self.get_pct()
    def nbasis(self):
        # return number of basis functions
        return len(self.basis)
    def state(self, Nr, ciDF=None):
        # Return the non-zero contributors to the SO state labeled 'Nr',
        # sorted by decreasing contribution.
        # the SO state 'Nr' is the column with index = Nr - 1
        # If 'ciDF' is provided, use it to obtain the term symbols
        # for the CI states.
        if Nr < 1:
            print('*** Error: SO state Nr must be >0')
            sys.exit(0)
        j = Nr - 1
        comp = self.basis.copy()
        comp['Pct'] = self.pct[:, j]
        grp = comp[comp.Pct > 0].groupby(['Spin', 'CI lbl'])
        ci = []
        p = []
        for lbl, g in grp:
            if ciDF is None:
                ci.append(lbl)
            else:
                # look for a term symbol
                symb = ciDF[(ciDF.Spin == lbl[0]) & (ciDF.Label == lbl[1])]['Term'].values[0]
                ci.append((lbl[0], lbl[1], symb))
            p.append(g['Pct'].sum())
        df = pd.DataFrame({'CI state': ci, 'Pct': p})
        return df.sort_values(by='Pct', ascending=False)
    def state_by_term(self, Nr, ciDF):
        # return the non-zero CI terms that contribute to SO state 'Nr'
        # sorted by decreasing contribution
        # relies upon term symbols being unique
        df = self.state(Nr, ciDF)
        # add column with term symbol
        terms = [row['CI state'][2] for i, row in df.iterrows()]
        df['Term'] = terms
        grp = df.groupby('Term').sum()
        return grp.sort_values(by='Pct', ascending=False)
    def get_basis(self):
        # extract the descriptors for the basis states
        rx_data = re.compile(r'(\s+\d+){4}')
        rx_int = re.compile(r'\d+')
        # different formatting for half-integer spin
        rx_alt = re.compile(r'\s+\d+(\s+\d+\.\d){2}\s*[-]?\d+\.\d')
        Nr = []
        mz = []
        cilbl = []
        mult = []
        for line in self.lines:
            if rx_data.match(line):
                words = line.split()
                n = int(words[0])
                if Nr and (n < max(Nr)):
                    # no new basis functions
                    break
                Nr.append(n)
                cilbl.append('.'.join(words[2:4]))
                # S is just a numerical part of the ket symbol
                s = rx_int.search(words[4]).group(0)
                mult.append(SPINMULT[float(s)])
                # Sz is more complicated
                m = int(rx_int.search(words[5]).group(0))
                if '-' in words[5]:
                    m = -m
                mz.append(m)
            if rx_alt.match(line):
                words = line.split()
                n = int(words[0])
                if Nr and (n < max(Nr)):
                    # no new basis functions
                    break
                Nr.append(n)
                cilbl.append(words[1])
                mult.append(SPINMULT[float(words[2])])
                mz.append(float(words[3]))
        # construct the descriptors (DataFrame)
        df = pd.DataFrame({'Nr': Nr, 'CI lbl': cilbl, 'Spin': mult, 'Sz': mz})
        return df.set_index('Nr')
    def get_pct(self):
        # return an array of the composition percentages
        # column = SO state
        # row = basis state
        rx_hdr = re.compile(r' Nr Sym  State Sym Spin ')
        rx_data = re.compile(r'(\s+\d+){4}')
        # half-integer output is formatted differently
        rx_alt = re.compile(r'  Nr  State  S   Sz ')
        matr = np.zeros((self.nbasis(), self.nbasis()))
        cols = []
        for line in self.lines:
            if rx_hdr.match(line):
                cols = line.split()[7:]
                ip = 6
            if rx_alt.match(line):
                cols = line.split()[4:]
                rx_data = re.compile(r'\s+\d+(\s+\d+\.\d){2}\s*[-]?\d+\.\d')
                ip = 4
            if rx_data.match(line):
                # remove the '%' symbols
                line = line.replace('%', '')
                words = line.split()
                for j, pct in zip(cols, words[ip:]):
                    k = int(j) - 1 # column index
                    i = int(words[0]) - 1  # row index
                    matr[i,k] = float(pct)
        return matr
    def printlines(self):
        print('\n'.join(self.lines))
##
class CASSCFstate():
    def __init__(self, mresult):
        # argument is one row (pd.Series) of a MULTI.results DataFrame
        self.label = mresult['Label']
        self.energy = mresult['Energy']
        words = self.label.split('.')
        self.number = int(words[0])
        self.irrep = int(words[1])
        try:
            # angular momentum 
            self.Lz = np.sqrt(abs(mresult['LzLz']))
        except:
            self.Lz = np.nan
        self.spin = MULTSPIN[mresult['Spin'].title()]  # number of unpaired electrons
    def set_lzlz(self, lzlz):
        # lzlz is a string
        lzlz = np.abs(float(lzlz))
        lz = np.sqrt(lzlz)
        self.Lz = int(round(lz))
        return
    def print(self, file=sys.stdout):
        L = LAMBDA[self.Lz]
        mult = SPINMULT[self.spin]
        print('CAS state {:s}: E = {:.6f}   {:s} {:s}'.format(self.label,
                self.energy, mult, L))
        return
    def equals(self, other):
        # does the other CASSCF/MRCI state have the same label and spin?
        retval = (self.label == other.label) and (self.spin == other.spin)
        return retval
    def descriptor(self):
        # a string that shows label, spin, and Lambda
        spinstr = halves(self.spin)
        retval = '({:s}, {:s}, {:s})'.format(self.label, spinstr, LAMBDA[self.Lz])
        return retval
    def omega_range(self):
        # return list of possible Omega values
        return omega_range(self.Lz, self.spin)
    def has_omega(self, omega):
        # Is Omega = 'omega' possible for this state?
        return omega in self.omega_range()
    def term_label(self, greek=False):
        # return label like '2-Delta' or (Greek) '2<Delta>'
        j = chem.round_half_int(self.Lz)
        if greek:
            L = GLAMBDA[j]
        else:
            L = LAMBDA[j].rstrip()
        mult = int(round(2 * self.spin + 1))
        return str(mult) + L
##
class MRCIstate(CASSCFstate):
    def __init__(self, mresult):
        # argument is one row of an MRCI.properties DataFrame
        CASSCFstate.__init__(self, mresult)
        self.reflabel = mresult['Ref']  # label for CAS state with max overlap
        self.dav = mresult['Edav']      # Davidson-corrected energy
        self.c0 = mresult['C0']        # coefficient of leading reference
    def set_reflabel(self, ovl):
        # set reference label to  state #ovl (same irrep)
        self.build_reflabel(int(ovl), self.irrep)
    def build_reflabel(self, state, irrep):
        self.reflabel = '{:d}.{:d}'.format(state, irrep)
    def print(self, file=sys.stdout):
        pstr = 'MRCI state ({:s}, {:s}): E = {:.6f}   Lz = {:.3f}  reflabel = {:s}'.format(self.label,
                halves(self.spin), self.energy, self.Lz, self.reflabel)
        pstr += '   Dav = {:.6f}   C0 = {:.2f}'.format(self.dav, self.c0)
        print(pstr)
##
##
def read_coordinates(fname, linenum=False):
    # return a DataFrame of coordinates
    # return a list of DF if there are multiple coordinates
    # If linenum == True, also return a list of corresponding line numbers
    rx_coord = re.compile('ATOMIC COORDINATES')
    rx_data = re.compile(r'^\s*\d+\s+[A-Z]+\s+\d+\.\d\d\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+')
    rx_blank = re.compile('^\s*$')
    dflist = []
    lineno = []
    cols = ['Z', 'x', 'y', 'z']
    incoord = False
    with open(fname, 'r', errors='replace') as F:
        for lno, line in enumerate(F):
            if incoord:
                if (rx_blank.match(line) and (len(df) > 0)):
                    incoord = False
                    dflist.append(df)
                if rx_data.match(line):
                    w = line.split()
                    row = [w[1], float(w[3]), float(w[4]), float(w[5])]
                    df.loc[len(df)] = row
            else:
                if rx_coord.search(line):
                    incoord = True
                    df = pd.DataFrame(columns=cols)
                    lineno.append(lno)
    if len(dflist) == 1:
        if linenum:
            return dflist[0], lineno[0]
        else:
            return dflist[0]
    else:
        if linenum:
            return dflist, lineno
        else:
            return dflist
##
def readMULTI(fname, linenum=False, PG=None):
    # return a list of MULTI objects
    # If linenum == True, also return a list of line numbers
    # 'PG' is the name of the point group (optional)
    rx_multi = re.compile(r'[ 1]PROGRAM \* MULTI \(Direct Multiconfiguration SCF\)')
    rx_end = re.compile(r' [*]{80}')
    retval = []
    casbuf = []
    lineno = []
    inMULTI = False
    with open(fname, 'r', errors='replace') as F:
        for lno, line in enumerate(F):
            if inMULTI:
                casbuf.append(line.rstrip())
                if rx_end.match(line):
                    inMULTI = False
                    # create the MULTI object
                    retval.append(MULTI(casbuf, PG=PG))
            if rx_multi.match(line):
                inMULTI = True
                casbuf = [line.rstrip()]
                lineno.append(lno)
    if linenum:
        return retval, lineno
    else:
        return retval
##
def readMRCI(fname, linenum=False):
    # return a list of MRCI objects
    # If linenum==True, also return a list of line numbers
    rx_ci = re.compile(r'[ 1]PROGRAM \* CI \(Multireference internally contracted CI\)')
    rx_so = re.compile(r'\*\*\* Spin-orbit calculation \*\*\*')
    rx_end = re.compile(r' [*]{80}')
    inci = False
    retval = []
    cibuf = []
    lineno = []
    with open(fname, 'r', errors='replace') as F:
        for lno, line in enumerate(F):
            if inci:
                cibuf.append(line.rstrip())
                if rx_so.search(line):
                    # This is SO-CI, a different case
                    inci = False
                    cibuf = []
                    lineno.pop()
                if rx_end.match(line):
                    inci = False
                    # create the MRCI object
                    retval.append(MRCI(cibuf))
            if rx_ci.match(line):
                inci = True
                cibuf = [line.rstrip()]
                lineno.append(lno)
    if linenum:
        return retval, lineno
    else:
        return retval
##
def combineMRCI(MRCIlist):
    # given a list of MRCI objects, return a DataFrame with all their info
    dtemp = []
    for i, ci in enumerate(MRCIlist):
        d = ci.results.copy()
        d.insert(0, 'Group', i+1)
        dtemp.append(d)
    df = pd.concat(dtemp, ignore_index=True)
    return df
##
def averageTerms(dfci, be_close=None, target=None, be_same=None, quiet=False):
    '''
    Given a DataFrame of MULTI or MRCI results, identify and average
      degenerate levels.
    Return a smaller DataFrame with (term) averages
    'be_close' is a list of quantities that must agree to 'target' * 10
    'target' is desire for how close 'be_close' items agree
      differences are acceptable up to 10 * 'target'
    'be_same' is a list of quantities that must be equal
    '''
    if be_close is None:
        # defaults appropriate for MRCI or MCSCF with term labels already assigned
        be_close = ['Energy', 'Edav', 'Dipole', 'LzLz', 'dipX', 'dipY', 'dipZ', 'Eref', 'C0']
        target =   [ 1.e-5,    1.e-5,  1.e-4,    1.e-4,  1.e-4,  1.e-4,  1.e-4,  1.e-4,  1.e-4]
    criteria = zip(be_close, target)
    if be_same is None:
        be_same = ['Spin', 'Term']
    dfret = pd.DataFrame(columns=dfci.columns)
    badcol = []  # list of columns to delete from the return DF
    for g, d in dfci.groupby(be_same):
        ds = d.sort_values(be_close[0])
        rowlist = [row for (i, row) in ds.iterrows()]
        while len(rowlist) > 0:
            n = len(rowlist)
            # compare all other rows with the first row
            used = [0]
            for j in range(1,n):
                ok = True
                for col, targ in criteria:
                    try:
                        diff = abs(rowlist[j][col] - rowlist[0][col])
                        if diff > 10*targ:
                            # not a match
                            ok = False
                            continue
                        if diff > targ:
                            if not quiet:
                                chem.printerr('Marginal agreement on {:s}'.format(col), halt=False)
                    except:
                        # probably missing column
                        pass
                    if not ok:
                        # stop checking 
                        break
                if ok:
                    # this row is a match
                    # mark any mismatching columns for deletion
                    for k, v in rowlist[0].iteritems():
                        if (k in be_close) or (k in be_same):
                            continue
                        if v != rowlist[j][k]:
                            badcol.append(k)
                    used.append(j)
            # average values
            for col in be_close:
                try:
                    rowlist[0][col] = np.mean([rowlist[j][col] for j in used])
                except:
                    # missing quantity
                    pass
            dfret = dfret.append(rowlist[0], ignore_index=True)
            # remove used rows from list
            for j in reversed(used):
                del rowlist[j]
        # remove bad columns
        dfret.drop(badcol, axis=1, inplace=True)
    return dfret
##
def readSOenergy(fname, recalc=False, linenum=False):
    # return an SOenergy object, or a list of them
    # if recalc==True, then re-calculate the wavenumber quantities from
    #    the hartree energies 
    # If linenum == True, also return line number(s)
    re_vals = re.compile(r' Eigenvalues of the spin-orbit matrix| Spin-orbit eigenstates')
    re_end = re.compile(r' E0 =\s+([-]?\d+\.\d+) is the energy of the lowest zeroth-order state')
    re_e0alt = re.compile(r' Lowest unperturbed energy E0=\s+([-]?\d+\.\d+)')
    re_endalt = re.compile(r' Eigenvectors of spin-orbit ')
    sobuf = []
    retlist = []
    lineno = []
    inso = False
    with open(fname, 'r', errors='replace') as F:
        for lno, line in enumerate(F):
            # output looks different depending upon J integer or half-integer
            m = re_e0alt.match(line)
            if m:
                E0 = float(m.group(1))
            if inso:
                sobuf.append(line.rstrip())
                m = re_end.match(line)
                if m:
                    E0 = float(m.group(1))
                    inso = False
                    retlist.append(SOenergy(sobuf, E0))
                    sobuf = []
                if re_endalt.match(line):
                    inso = False
                    retlist.append(SOenergy(sobuf, E0))
                    sobuf = []
            if re_vals.match(line):
                inso = True
                sobuf = [line.rstrip()]
                lineno.append(lno)
    if recalc:
        for so in retlist:
            so.recalc_wavenumbers()
    if len(retlist) == 1:
        if linenum:
            return retlist[0], lineno[0]
        else:
            return retlist[0]
    else:
        if linenum:
            return retlist, lineno
        else:
            return retlist
##
def readSOcompos(fname):
    # return an SOcompos object
    # assume there is only one SO-CI calculation
    rx_socomp = re.compile(r' Composition of spin-orbit eigenvectors')
    rx_end = re.compile(r' [*]{80}|  Expectation values | Property matrices transformed')
    buf = []
    incomp = False
    with open(fname, 'r', errors='replace') as F:
        for line in F:
            if incomp:
                buf.append(line.rstrip())
                if rx_end.match(line):
                    incomp = False
                    break
            if rx_socomp.match(line):
                incomp = True
                buf = [line.rstrip()]
    return SOcompos(buf)
##
def parse_SOmatrix(buf, dimen, assist=None, silent=False):
    # return the S-O matrix (cm-1) as a numpy array
    # also return a list of state labels for the basis states
    #    state labels are tuples (State, S, Sz) as in the MOLRPO file
    # State labels may be wrong and require correction
    #   then you should supply 'assist' as a list of tuples (label, spin)
    rx_header = re.compile(r'\s+Nr\s+State\s+S\s+S[Zz](\s+\d+)+')
    rx_real = re.compile(r'^\s+\d+\s+\d+\.\d\s+\d+\.\d\s+[-]?\d+\.\d(\s+[-]?\d+\.\d+)+$')
    rx_imag = re.compile(r'^(\s+[-]?\d+\.\d+)+$')
    #
    basis = []  # tuples describing the basis states (str, float, float)
    blockbasis = []  # as above, but within one 'Nr State' block (to detect label error)
    mat = np.zeros((dimen, dimen), dtype=complex)
    for line in buf:
        if rx_header.match(line):
            words = line.split()
            cols = [int(x)-1 for x in words[4:]]
            blockbasis = []
        if rx_real.match(line):
            words = line.split()
            tup = (words[1], float(words[2]), float(words[3]))
            if tup in blockbasis:
                if not silent:
                    chem.print_err('', 'Probable CI state labeling error: {:s}'.format(str(tup)), halt=False)
                if assist is not None:
                    # try to correct the label
                    soon = False
                    for ilbl, lbl in enumerate(assist):
                        if soon:
                            if tup[1] == lbl[1]:
                                # this is the right spin
                                if tup[0][-1] == lbl[0][-1]:
                                    # this is the right irrep
                                    if tup[0].split('.')[0] != lbl[0].split('.')[0]:
                                        # different from the wrong label--choose this 
                                        tup = (lbl[0], tup[1], tup[2])
                                        if not silent:
                                            print('\tlabel corrected to {:s} based upon "assist" list'.format(tup[0]))
                                        break
                        if (tup[0] == lbl[0]) and (tup[1] == lbl[1]):
                            # match for the incorrect label
                            soon = True  # start looking
            blockbasis.append(tup)
            if tup not in basis:
                basis.append(tup)
            irow = int(words[0]) - 1
            for icol, elem in zip(cols, words[4:]):
                mat[irow, icol] = float(elem)
        if rx_imag.match(line):
            words = line.split()
            for icol, elem in zip(cols, words):
                mat[irow, icol] += float(elem) * 1j
    return mat, basis
##
def readSOmatrix(fname, dimen, assist=None, silent=False):
    # return the S-O matrix and the list of basis states
    # if there is more than one SO-CI calculation, return a list
    rx_energ = re.compile(r' Spin-orbit eigenstates ')
    rx_somat = re.compile(r' Spin-Orbit Matrix \(CM-1\)')
    rx_symm_ad = re.compile(r'Spin-orbit calculation in the basis of symmetry adapted')
    somatbuf = []
    in_somat = False
    retval = []
    with open(fname, 'r', errors='replace') as F:
        for line in F:
            if in_somat:
                if rx_energ.match(line) or rx_symm_ad.search(line):
                    in_somat = False
                    retval.append(parse_SOmatrix(somatbuf, dimen, assist=assist, silent=silent))
                    somatbuf = []
                    continue
                somatbuf.append(line)
            if rx_somat.match(line):
                in_somat = True
    if len(retval) == 1:
        # do not return a list
        return retval[0]
    else:
        return retval
##
def get_SOeigs(SOmatrix):
    # Diagonalize and sort by increasing energy
    # check that eigenvalues are real
    # Return eigenvalues and squares of eigenvector elements
    # In array vecsq, the first index is the basis state and the 
    #   second index is the eigenvalue
    eigvals, eigvecs = np.linalg.eig(SOmatrix)
    for eig in eigvals:
        if abs(eig.imag) > 0.01:
            print('** eigenvalue has imaginary component: {:.2f} +'.format(eig.real),
                  'i*', eig.imag)
    vals = eigvals.real  # discard imaginary parts, which should be zero
    vecsq = np.abs(eigvecs) ** 2 # squares of eigenvector elements (c**2)
    # sort by increasing energy
    idx = np.argsort(vals)
    vals = vals[idx]
    vecsq = vecsq[:, idx]
    return vals, vecsq
##
def applied_field(fname):
    # return the applied electric field vector (atomic units)
    rx_field = re.compile(' Field strength:(\s+[-]?\d+\.\d+){3}')
    field = np.zeros(3)
    with open(fname, 'r', errors='replace') as F:
        for line in F:
            if rx_field.match(line):
                words = line.split()
                field = np.array(words[2:], dtype=float)
                break
    return field
##
def collect_degenerate(df, cmtol=1.):
    # for use with SOenergy().energies DataFrames
    # return a DataFrame with degenerate levels averaged and counted
    # levels are "degenerate" when their differences <= 'cmtol', in cm-1 units
    #    "degenerate" is transitive, so differences may exceed the tolerance
    # Ignore column "Irrep"
    lE = []      # list of energies 'E' (hartree)
    lshift = []  # list of energies 'Eshift' (cm-1)
    lrel = []    # list of relative energies 'Erel' (cm-1)
    count = []   # multiplicity of each grouping
    idxlist = [] # list of lists of indices for energy groupings
    e = None
    for irow, row in df.sort_values('E').iterrows():
        if e is None:
            # initialize a grouping
            e = row['Erel']
            idx = [row.name]
            continue
        de = row['Erel'] - e  # non-negative because 'df' has been sorted
        if de <= cmtol:
            # add to current grouping
            idx.append(row.name)
            e = row['Erel']   # for transitivity
        else:
            # close out the grouping
            lE.append(df.loc[idx]['E'].mean())
            lshift.append(df.loc[idx]['Eshift'].mean())
            lrel.append(df.loc[idx]['Erel'].mean())
            count.append(len(idx))
            idxlist.append(idx)
            # start a new grouping
            e = row['Erel']
            idx = [row.name]
    # close out the last grouping
    lE.append(df.loc[idx]['E'].mean())
    lshift.append(df.loc[idx]['Eshift'].mean())
    lrel.append(df.loc[idx]['Erel'].mean())
    count.append(len(idx))
    idxlist.append(idx)
    # summarize in new DataFrame
    cols = ['degen', 'E', 'Eshift', 'Erel', 'index']
    data = zip(count, lE, lshift, lrel, idxlist)
    dfcoll = pd.DataFrame(data=data, columns=cols)
    return dfcoll
##
def termLabels(casDF, greek=True, hyphen=False, PG=None):
    '''
    Given a DataFrame with a column labeld 'Spin' and a column 
      labeled either 'L**2' (for an atom) or 'LzLz' (for a linear
      molecule), return another DataFrame with an added column
      'Term'.
    If greek=False, Greek letters (for linear molecules) will
      be replaced by words.  E.g. 'Delta' instead of the character.
    If hyphen==True, put a hyphen between the spin and the L label.
    ''' 
    dfnew = casDF.copy()
    isatom = 'L**2' in dfnew.columns
    lbl = []
    if isatom:
        for lsq in casDF['L**2']:
            x = np.sqrt(4*lsq + 1) - 1
            Lval = int(np.round(x/2))
            lbl.append(LSYMB[Lval])
    else:
        # linear molecule
        for irow, lzsq in enumerate(casDF['LzLz']):
            Lval = int(np.round(np.sqrt(lzsq)))
            if greek:
                lbl.append(GLAMBDA[Lval])
            else:
                lbl.append(LAMBDA[Lval])
            '''
            if Lval == 0:
                # Sigma state: '+' or '-' ?
                par = ''
                print(f'>>>irrep = {casDF.iloc[irow].Irrep}, spin = {casDF.iloc[irow].Spin}')
                if PG.lower() == 'c2v':
                    # assign '+' to irrep #1
                    if casDF.iloc[irow].Irrep == 1:
                        par = '+'
                    else:
                        # it should be irrep #4 (a2)
                        # is it an open-shell singlet?
                        par = '-'
                lbl[-1] += par
                print(f'\tpar = {par}, lbl[-1] = {lbl[-1]}')
            '''
    # lbl[] describes the orbital angular momenta
    # prepend the spin multiplicities
    symb = []
    for spin, orb in zip(casDF['Spin'], lbl):
        # Expect either value of S (a multiple of 0.5) or
        # a string like 'Singlet'
        try:
            mult = 2*spin + 1.0
        except:
            # convert string to number
            x = MULTSPIN[spin]
            mult = 2*x + 1.0
        if hyphen:
            symb.append('{:.0f}-{:s}'.format(mult, orb))
        else:
            symb.append('{:.0f}{:s}'.format(mult, orb))
    dfnew['Term'] = symb
    return dfnew
##
def readTable(fname, title):
    # return a DataFrame of the contents of a MOLPRO table
    # table must have a title (2nd argument here)
    # the 'title' should be the complete title and work inside a regular expression
    rx_title = re.compile('^\s*{:s}\s*$'.format(title))
    rx_blank = re.compile(r'^\s*$')
    in_tbl = Ffalse
    buf = []
    with open(fname, 'r', errors='replace') as F:
        for line in F:
            if in_tbl:
                buf.append(line.rstrip())
            if rx_blank.match(line) and (len(buf) > 4):
                # done reading this table
                break
            if rx_title.match(line):
                buf = [line.rstrip()]
                # expect one blank line to follow
                in_tbl = True
    # parse the table
    cols = []
    for i in range(2, len(buf)-1):
        if not cols:
            # read the column headings
            cols = buf[i].split()
            data = [[] for col in cols]  # list of lists
        else:
            # read a line of data, attempt to convert to float
            words = buf[i].split()
            for i, word in enumerate(words):
                try:
                    data[i].append(float(word))
                except:
                    data[i].append(word)
    try:
        df = pd.DataFrame({col: vec for col, vec in zip(cols, data)})
    except:
        # probably did not find the table
        df = pd.DataFrame()
    return df
##
##
def halves(spin):
    # n = 2*spin is rounded to the nearest integer
    # return a string that looks like n/2 when
    #   n is odd and just the integer n/2 when even
    # E.g. 3 -> '3/2' and 4 -> '2'
    n = int(round(2*spin))
    if n % 2:
        # spin is odd
        spinstr = '{:d}/2'.format(n)
    else:
        spinstr = '{:d}'.format(n // 2)
    return spinstr
##
def omega_range(L, spin, rounding=True):
    # return the list of possible omega values for this Lambda, Sigma
    S = spin
    lo = L - S
    hi = L + S
    vals = np.append(np.arange(lo, hi), hi)
    vals = np.abs(vals)
    if rounding:
        # round to nearest half-integer
        vals = chem.round_half_int(vals)
    return vals
##
def omega_counts(cas, silent=False, rounding=True):
    # Return a dict of Omega values spanned by the CASSCF or MRCI states
    omlist = []
    for c in cas:
        omlist.extend(c.omega_range())
    if rounding:
        # round omega values to nearest 0.5
        omlist = chem.round_half_int(omlist).tolist()
    omegavals = chem.list_counts(omlist)
    if not silent:
        print('Omega counts from CASSCF/MRCI:')
        print('{:s}\t#'.format(OMEGA))
        ocount = 0
        for k in sorted(omegavals.keys()):
            print('{:.1f}\t{:d}'.format(k, omegavals[k]))
            ocount += omegavals[k]
        print('Total of {:d} states'.format(ocount))
    return omegavals
##
def read_point_group(fname):
    # get the computational point group
    rx_PG = re.compile(r' Point group\s+(\S+)\s*$')
    with open(fname, 'r') as F:
        for line in F:
            m = rx_PG.match(line)
            if m:
                PG = m.group(1)
                return PG
    # found nothing
    return None
##
def irreps(PG):
    # given the name of a point group, return two lists taken from the
    # Molpro manual (index = irrep # and starts from 1)
    if PG is None:
        return [None], [None]
    pg = PG.lower()
    if pg == 'c1':
        name = [None, 'A']
        function = [None, ('s', 'x', 'y', 'z', 'xy', 'xz', 'yz')]
    elif pg == 'cs':
        name = [None, 'A\'', 'A"']
        function = [None, ('s', 'x', 'y','xy'), ('z', 'xz', 'yz')]
    elif pg == 'c2':
        name = [None, 'A', 'B']
        function = [None, ('s', 'z', 'xy'), ('x', 'y', 'xz', 'yz')]
    elif pg == 'ci':
        name = [None, 'Ag', 'Au']
        function = [None, ('s', 'xy', 'xz', 'yz'), ('x', 'y', 'z')]
    elif pg == 'c2v':
        name = [None, 'A1', 'B1', 'B2', 'A2']
        function = [None, ('s','z'), ('x','xz'), ('y','yz'), ('xy',)]
    elif pg == 'c2h':
        name = [None, 'Ag', 'Au', 'Bu', 'Bg']
        function = [None, ('s','xy'), ('z',), ('x','y'), ('xz', 'yz')]
    elif pg == 'd2':
        name = [None, 'A', 'B3', 'B2', 'B1']
        function = [None, ('s',), ('x','yz'), ('y','xz'), ('xy',)]
    elif pg == 'd2h':
        name = [None, 'Ag', 'B3u', 'B2u', 'B1g', 'B1u', 'B2g', 'B3g', 'Au']
        function = [None, ('s',), ('x',), ('y',), ('xy',), ('z',),
                    ('xz',), ('yz',), ('xyz',)]
    else:
        chem.printerr('', 'Unknown point group {:s}'.format(PG))
    return name, function
##
def match_spin_label(lbl, S, mrci):
    '''
    Given a label (like '3.1') and a spin (multiple of 0.5),
    and a list of CASSCFstate() objects, return a list of indices into
    that list for state that match both label and spin. 
    '''
    idx = []
    for i, m in enumerate(mrci):
        if (m.label == lbl) and (m.spin == S):
            idx.append(i)
    return idx
##
def compare_MOLPRO_SOvals(energcm, vals, silent=False):
    # Compare my energies ('vals'), from diagonalizing the SO matrix,
    # with those reported by Molpro ('energcm')
    # if silent==True, print nothing
    # Return the largest difference
    maxdiff = 0
    if not silent:
        print('Eigenvalues from MOLPRO and here:')
        print('{:3s}  {:>8s}  {:>8s}  {:>8s}'.format('#', 'MOLPRO', 'here', 'diff'))
    eigs = vals.tolist()
    for mpro, eig in zip(energcm, eigs):
        diff = eig - mpro
        if abs(diff) > abs(maxdiff):
            maxdiff = diff
        if not silent:
            print('{:3d}  {:8.2f}  {:8.2f}  {:8.2f}'.format(eigs.index(eig), mpro, eig, diff))
    if not silent:
        print('Biggest diff = {:.2f}'.format(maxdiff))
    return maxdiff
##
def link_MRCI_SObasis(mrci, SObasis):
    '''
    connect SO basis states with their MRCI parents
    'mrci' is a list of MRCIstate objects
    'SObasis' is a list of tuples that label SO basis states (CI_lbl, S, M_S)
    return two lists:  
      sob_ici (index into mrci[] for each SO basis state)
      ci_sob (list of basis states derived from each mrci state)
    '''
    sob_ici = [] 
    ci_sob = [[] for m in mrci] 
    for iso, b in enumerate(SObasis):
        lbl = b[0]  # e.g., '3.1'
        S = b[1]    # spin
        ici = match_spin_label(lbl, S, mrci)
        if len(ici) != 1:
            errstr = '**** Ambiguous or missing MRCI assignment for SO basis state:\n'
            errstr += str(b, ici)
            chem.printerr('', errstr)
        ici = ici[0]
        sob_ici.append(ici)
        try:
            ci_sob[ici].append(iso)
        except:
            # create element (a list)
            ci_sob[ici] = [iso]
    for i, c in enumerate(ci_sob):
        if len(c) == 0:
            s = f'CI state {i} is not represented in the SO basis\n'
            s += 'check your Molpro file for duplicate state numbering in the SO basis'
            chem.print_err('', s)
    return sob_ici, ci_sob
##
def assign_omega_possibilities(mrci, SObasis, vecsq, csq_thresh=0.0001, silent=True):
    '''
    Assign possible Omega values to spin-orbit states
    Return: a list of sets of possible Omega values
            an array of c**2 values for MRCI states (sum over component SO basis states)
    'mrci' is a list of MRCIstate()
    'SObasis' is a list of tuples describing SO basis states (CI-lbl, S, MS)
    'vecsq' is from diagonalizing the SO matrix using get_SOeigs()
    'csq_thresh' is the threshold for discarding contributions
    '''
    sob_ici, ci_sob = link_MRCI_SObasis(mrci, SObasis)
    om_possible = []  # list of sets of possible Omega values
    dimen = vecsq.shape[1]
    if not silent:
        print('--- Find possible {:s} values for each SO state ---'.format(OMEGA))
    for istate in range(dimen):
        if not silent:
            print(f'\tstate {istate}')
        idx = np.argwhere(vecsq[:,istate] > csq_thresh).flatten()
        poss = [] # list of sets
        for i in idx:
            ici = sob_ici[i]  # index of parent MRCI state
            m = mrci[ici]     # parent MRCIstate
            s = SObasis[i]
            spinOK = m.spin == s[1]
            x = set((np.round(abs(s[2]-m.Lz), 1), np.round(abs(s[2]+m.Lz), 1)))
            if spinOK:
                poss.append(x)
            else:
                print('*** Inconsistent spins! istate={:d}, ici={:d}, SO basis S = {:.1f}, CI S = {:.1f}'.format(istate,
                        ici, s[1], m.spin))
        possib = set.intersection(*poss)
        if not silent:
            print(f'\t\t{poss}')
            print(f'\t\treduces to {possib}')
        if len(possib) == 0:
            msg = '*** No possible omega values for istate={:d}! Try increasing csq_thresh'
            print(msg)
        om_possible.append(possib)
    # sums within MRCI states
    ciwt = np.zeros((len(mrci), dimen))
    for i in range(len(mrci)):
        for j in ci_sob[i]:
            ciwt[i, :] = ciwt[i, :] + vecsq[j, :]
    return om_possible, ciwt
##
def assign_Omega_values(mrci, SObasis, vals, vecsq, csq_thresh=0.0001, silent=False):
    '''
    Assign values of Omega to SO-CI states
    'mrci' is a list of MRCIstate()
    'SObasis' is a list of tuples describing SO basis states (CI-lbl, S, MS)
    'vecsq' is from diagonalizing the SO matrix using get_SOeigs()
    'csq_thresh' is the threshold for discarding contributions
    Return: DataFrame that includes energies, Omegas, term labels, etc.
    '''
    if not silent:
        print('*** Assign Omega values using weight threshold of', csq_thresh, '***')
    # casscf/mrci detrmines the Omega values
    omegavals = omega_counts(mrci, silent=silent)
    dimen = vecsq.shape[1]
    om_assigned = [np.nan] * dimen
    unassigned = omegavals.copy()
    omega_left = [m.omega_range().tolist() for m in mrci]  # possible remaining Omegas for each MRCI state
    nassigned = 0
    om_possible, ciwt = assign_omega_possibilities(mrci, SObasis, vecsq, csq_thresh=csq_thresh,
                                                   silent=silent)
    # find Omega possibilites for each MRCI state
    ciOm = [set(m.omega_range()) for m in mrci]

    def assign_state(istate, imrci, omega):
        # modify nonlocal vars
        nonlocal om_assigned, om_possible, omega_left, unassigned
        # if imrci == 'all', sweep this value of omega
        om_assigned[istate] = omega   # the actual assignment
        om_possible[istate] = set()   # no update is possible because the state is now assigned
        if imrci == 'all':
            #print('\t\tassigning {:s} = {} to state {:d} from unspecified MRCI'.format(mpr.OMEGA, omega, istate))
            unassigned[omega] = 0
            for left in omega_left:
                while omega in left:
                    left.remove(omega)
        else:
            #print('\t\tassigning {:s} = {} to state {:d} from MRCI {:d}'.format(mpr.OMEGA, omega, istate, imrci))
            unassigned[omega] -= 1
            omega_left[imrci].remove(omega)  # remove one instance of this omega value
        return

    def step1():
        # Find any MRCI states with only one possible Omega
        #   then assign that value of Omega by decreasing weight, not to exceed
        #   the maximum listed in omegavals{}
        nonlocal omegavals, ciOm, omega_left, mrci, ciwt, csq_thresh
        if not silent:
            print('Step 1 of assignments')
        n1 = 0
        for om, count in omegavals.items():
            # make list of MRCI states that allow only Omega = om
            cistates = []  # list of the MRCI states
            cilist = []    # similar but accounting for mulitiple occurences of 'om'
            for ici, allowed in enumerate(ciOm):
                if (om in allowed) and (len(allowed) == 1):
                    cistates.append(ici)
                    cilist.extend([ici] * len(omega_left[ici]))
                    if not silent:
                        print('\tMRCI state #{:d} ({:s}) must have {:s} = {}'.format(ici+1, 
                                            mrci[ici].term_label(greek=True), OMEGA, om))
            if len(cistates) > 0:
                # make a table of the weights from these states
                wts = np.zeros((len(cistates), dimen))
                for i, ici in enumerate(cistates):
                    wts[i, :] = ciwt[ici, :]
                # sum the weights across MRCI contributions
                wtsum = wts.sum(axis=0)
                idx = np.argsort(-wtsum)  # decreasing order
                # the number of states to assign is the lesser of:
                #   - the number allowed
                #   - the number of MRCI contributions
                n = min(omegavals[om], len(cilist))
                # truncate idx[] to the allowed length
                idx = idx[:n]
                #print('\tAssigning state(s) to {:s} = {}:'.format(OMEGA, om), idx)
                for i in idx:
                    # record the state assignments (Omega values)
                    if not np.isnan(om_assigned[i]):
                        chem.print_err('','*** Error: state {:d} is already assigned!'.format(i))
                    elif wtsum[i] > csq_thresh:
                        # assign Omega value to SO state
                        ici = cilist.pop()
                        assign_state(i, ici, om)
                        n1 += 1
                    else:
                        chem.print_err('','*** Error: required weight of {:.2e} is less than requested csq_thresh ({:.1e})'.format(wtsum[i],
                                csq_thresh))
        if not silent:
            print('\t{:d} states assigned'.format(n1))
        return n1
    
    def step2():
        # Is there an Omega for which the exact number of states match?
        nonlocal unassigned, om_possible
        if not silent:
            print('Step 2 of assignments')
        n2 = 0
        for om, count in unassigned.items():
            # how many states can have Omega == om?
            isposs = [(om in poss) for poss in om_possible]
            nposs = sum(isposs)
            if count > nposs:
                chem.print_err('',('For {:s} = {:.1f}, there are {:d} possible'
                          ' assignments but {:d} are needed').format(OMEGA,
                        om, nposs, count))
            if count == nposs:
                # Exact match; assign these states
                if not silent:
                    print('\tThe number of candidates for {:s} = {} equals the number needed'.format(OMEGA, om))
                    #print('\tAssigning state(s) to {:s} = {}:'.format(OMEGA, om), np.where(isposs)[0])
                for i, poss in enumerate(om_possible):
                    if om in poss:
                        assign_state(i, 'all', om)
                        n2 += 1
        if not silent:
            print('\t{:d} states assigned'.format(n2))
        return n2
    
    def step3():
        # Find states with only one possible assignment 
        nonlocal unassigned, om_possible, ciwt, omega_left
        if not silent:
            print('Step 3 of assignments')
        n3 = 0
        for om, count in unassigned.items():
            idx = []
            for i, poss in enumerate(om_possible):
                if (om in poss) and (len(poss) == 1):
                    # only this assignment is possible
                    idx.append(i)
            if idx:
                if not silent:
                    print('\tState(s) that can only have {:s} = {}:'.format(OMEGA, om), np.array(idx)+1)
                # for each MRCI state, sum its contributions to these states
                wtsum = ciwt[:, idx].sum(axis=1)
                for ici in np.argsort(-wtsum):
                    # decreasing total contribution
                    while len(idx) and (om in omega_left[ici]):
                        i = idx.pop(0)
                        assign_state(i, ici, om)
                        n3 += 1
                    if len(idx) == 0:
                        # all done here
                        break
                else:
                    # did not find a suitable MRCI parent
                    chem.print_err('','*** Error: no suitable parent MRCI found for {:s} = {:.1f}, state Nr {:d}'.format(OMEGA,
                            om, i+1))
            # is this Omega fully assigned?
            left = unassigned[om]
            if left == 0:
                # remove from sets of state possibilities
                for s in om_possible:
                    s.discard(om)
        if not silent:
            print('\t{:d} states assigned'.format(n3))
        return n3

    nassigned = step1()
    n2 = step2()
    nassigned += n2
    if not silent:
        print('Total of {:d} states assigned'.format(nassigned))
    while True:
        n3 = step3()
        nassigned += n3
        if not silent:
            print('Total of {:d} states assigned'.format(nassigned))
        if (nassigned == dimen) or (n3 == 0):
            break
    if (nassigned == dimen):
        if not silent:
            print('All states have been assigned!')
    else:
        # failure
        print('*** {:d} states remain unassigned!'.format(dimen - nassigned))
        chem.print_err('', 'Try more iterations of steps 1,2,3?')

    # prepare DF of state information; take term symbol from largest MRCI contributor
    term_assigned = []
    for j in range(dimen):
        ilead = np.argmax(ciwt[:, j])
        term_assigned.append(mrci[ilead].term_label(greek=True))
    # Prepare table for display
    labels = []
    for i in range(dimen):
        lbl = term_assigned[i] + '_' + halves(om_assigned[i])
        labels.append(lbl)
    dffinal = pd.DataFrame({'cm-1': np.round(vals, 1), OMEGA: om_assigned,
                            'term': term_assigned, 'label': labels})
    dffinal['exc'] = dffinal['cm-1'] - min(dffinal['cm-1'])
    return dffinal
##