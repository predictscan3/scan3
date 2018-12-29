from scan3.server.data_import import binary_norm
from scan3.server.data_import import ethnicity_norm


def apply_normalisation(df=None, save=False):
    df_binary = binary_norm.apply_binary_norm(df)
    df_ethnic = ethnicity_norm.apply_normalisation(df_binary)

    return df_ethnic
