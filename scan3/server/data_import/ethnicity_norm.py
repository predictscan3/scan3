from os.path import join, dirname
import pandas as pd
import numpy as np
import re
import json
from scan3 import settings


# This helps us to figure out where we need to look at both fields to help figure out the ethnic group
MISSING_ETHNIC_VALUES = ("missing",
                         "not specified", "not stated", "not given",
                         "patient unwilling to disclose")

OTHER_ETHNIC_VALUES = ("other", "other mixed", "mixed", "mixed other", "other mixed race",
                       "any other group", "other ethnic group", "mixed ethnic group")

ETHNIC_FIELDS = ["dem_ethnic_group", "dem_ethnic_group2"]


def make_mapper(target_name, synonyms):
    def scan_synonyms(fname, val):
        if isinstance(val, float) and np.isnan(val):
            return target_name
        else:
            # Remove annoying characters and collapse multiple spaces
            val = re.sub(" {2,10}", " ",
                         re.sub("and|backgrou|back gro|back ground| und| unspecif", "",
                                re.sub("wb", "white british",
                                       re.sub("[_/()-]", " ", val.strip().lower()))))
            if val is None or val in synonyms:
                return target_name
            else:
                return val

    return scan_synonyms


def tidy_missing_other(df):
    """
    Sort out missing values, so we can make it easier to process and analyse this data
    """
    map_missing = make_mapper("missing", MISSING_ETHNIC_VALUES)
    map_other = make_mapper("other", OTHER_ETHNIC_VALUES)

    for fname in ETHNIC_FIELDS:
        df[fname] = df[fname].map(lambda x: map_missing(fname, x))
        df[fname] = df[fname].map(lambda x: map_other(fname, x))

    return df


def apply_normalisation(df=None, save=False):
    df = tidy_missing_other(df)

    # Load the mapping file, and apply the normalisation
    df["dem_ethnic_key"] = df.dem_ethnic_group + " | " + df.dem_ethnic_group2
    map_file = join(settings.DATA_IN_ROOT, "ethnicity_map.json")
    with open(map_file, "rb") as f:
        ethnic_map = json.load(f)

    def ethnic_mapper(k):
        return ethnic_map.get(k, "unknown")

    df["dem_ethnic_norm"] = df.dem_ethnic_key.map(ethnic_mapper)

    return df


def generate_report(df):
    # cdfs = {}
    # for fname in ETHNIC_FIELDS:
    #     vals = list(set(df[fname]))
    #     counts = [len(df[df[fname] == val] == True) for val in vals]
    #     count_df = pd.DataFrame({"count": counts, "pct": np.array(counts) / float(len(df))}, index=vals)
    #     count_df.sort_values(by="count", inplace=True, ascending=False)
    #     cdfs[fname] = count_df
    #
    # return cdfs
    report = {}

    for k, sdf in df.groupby("dem_ethnic_norm"):
        report[k] = "{:,}, {:.0%}".format(len(sdf), float(len(sdf)) / len(df))

    return report

