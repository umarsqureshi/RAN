"""Observable definitions for 6D jet substructure analysis."""

OBS_NAMES = ['m', 'M', 'w', 'tau21', 'zg', 'sdm']

OBS_LABELS = {
    'm': r'$m$ [GeV]',
    'M': r'$M$',
    'w': r'$w$',
    'tau21': r'$\tau_{21}^{(\beta=1)}$',
    'zg': r'$z_g$',
    'sdm': r'$\ln \rho$',
}

OBS_TITLES = {
    'm': 'Jet Mass',
    'M': 'Jet Constituent Multiplicity',
    'w': 'Jet Width',
    'tau21': 'N-subjettiness Ratio',
    'zg': 'Groomed Jet Momentum Fraction',
    'sdm': 'Soft Drop Jet Mass',
}
