import numpy as np

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print big_string
    return np.nan

title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
              'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme',
              'Countess', 'Don', 'Jonkheer']

Names = ['Mrs Long', 'Mr Hua', 'Master Hua', 'Miss Long', 'Hello']
print map(substrings_in_string, Names, title_list)
