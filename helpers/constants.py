# Original Targets
TARGET_capacity = "Capacity(mAh/g)" # Chosen for diffusion model guide, but can be changed to any of the three targets
TARGET_capacity_retention = "Capacity Retention(%)"
TARGET_first_coulombic_eff = "First Coulombic Efficiency(%)"

EXPERIMENTAL_CAPACITY_RANGE = (200, 290)

# Model parameters
EMBEDDING_DIMENSION = 128

# Ranges provided by experimental group
CONTINUOUS_RANGE = {
    'Fraction_Li': (0.1, 2.0),
    'Fraction_O': (0.0, 5.0),
    'Fraction_Mg': (0.0, 1.0),
    'Fraction_Al': (0.0, 1.0),
    'Fraction_Ti': (0.0, 1.0),
    'Fraction_Mn': (0.0, 2.0),
    'Fraction_Co': (0.0, 1.0),
    'Fraction_Ni': (0.0, 2.0),
    'Fraction_Sr': (0.0, 0.1),
    'Fraction_Nb': (0.0, 1.0),
    'Fraction_Mo': (0.0, 1.0),
    'Fraction_Sb': (0.0, 0.1),
    'Fraction_Ta': (0.0, 0.5),
    'Fraction_W': (0.0, 0.1),
    'Particle Size Primary(nm)': (100, 2000),
    'Particle Size Secondary(nm)': (5000, 18000),
    'Annealing Temperature 1(K)': (725, 925),
    'Annealing Temperature 2(K)': (925, 1175),
    'Annealing Time 1(h)': (3, 6),
    'Annealing Time 2(h)': (10, 24),
    'Minimum Voltage(V)': (1.0, 3.5),
    'Maximum Voltage(V)': (3.7, 5.0),
    'Cycles': (5, 1000)
}
