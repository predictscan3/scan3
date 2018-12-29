import numpy as np
from collections import defaultdict


BINARY_FIELD_MAPS = dict(
    dem_alcohol={
        "n/k": None,
        "no alcohol": 0,
        "alcohol": 1
    },
    dem_cigarettes={
        "no": 0,
        "smoker": 1
    }
)


def binary_norm_field(fname, fval):
    """
    Normalise a particular field and value
    :param fname:
    :param fval:
    :return:
    """
    mapper = BINARY_FIELD_MAPS[fname]
    if isinstance(fval, float) and np.isnan(fval):
        return None
    try:
        return mapper[fval.lower()]
    except KeyError:
        raise KeyError("No {} mapping for {}".format(fname, fval))


def get_binary_counts(df, fname):
    """
    Generate a report of normed values and counts
    :param df:
    :param fname:
    :return:
    """
    vals = ("nan", 0., 1.)
    counts = []
    pcts = []

    for val in vals:
        if val == "nan":
            counts.append(len(df[df[fname].map(np.isnan) == True]))
        else:
            counts.append(len(df[(df[fname] == val) == True]))
        pcts.append(counts[-1] / float(len(df)))

    return zip(vals, pcts, counts)


def apply_binary_norm(df=None):
    """
    Apply the normalisations to the passed DataFrame
    :param df:
    :return:
    """

    for fname in BINARY_FIELD_MAPS.keys():
        normed_name = "{}_norm".format(fname)
        df[normed_name] = df[fname].map(lambda x: binary_norm_field(fname, x))

    return df


def generate_report(df=None):
    report = defaultdict(dict)
    
    for fname in BINARY_FIELD_MAPS.keys():
        normed_name = "{}_norm".format(fname)

        for v in get_binary_counts(df, normed_name):
            report[normed_name][v[0]] = "{} {:.0%} {:.0f}".format(*v)

    return dict(report)
