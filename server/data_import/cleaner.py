import pandas as pd
import settings
from os.path import join, exists, splitext
from os import makedirs
import cPickle as pickle
import json
from collections import OrderedDict
from re import sub

from scan3.server.data_import.joiner import join_center_scans


PARENT_FIELDS = [
    "patients_id",
    "dob",
    "ethnic_group2",
    "ethnic_group",
    "date_of_exam",
    "maternal_age_at_exam",
    "episode_lmp",
    "conception",
    "weight_kg",
    "height_cm",
    "cigarettes",
    "alcohol",
    "para",
    "iud_lt_15w",
    "iud_16_23w",
    "iud_24_36w",
    "iud_gte_37w",
    "crl_1",
    "nt1",
    "bpd3",
    "us_gestation_weeks1",
    "days1",
    "1st_trim_msb_date",
    "msb_manufacturer",
    "b_hcg",
    "_bhcg_MoM",
    "pappa",
    "pappa_MoM",
    "msb_plgf_pgml",
    "msb_plgf_MoM",
    "r_uterine_a_pi",
    "r_uterine_a_ri",
    "l_uterine_a_pi",
    "l_uterine_a_ri"
]

BABY_FIELDS = [
    "patients_id",
    "dob",
    "ethnic_group2",
    "ethnic_group",
    "date_of_exam",
    "maternal_age_at_exam",
    "episode_lmp",
    "conception",
    "edd_lmp",
    "edd_us",
    "weight_kg",
    "height_cm",
    "cigarettes",
    "alcohol",
    "para",
    "iud_lt_15w",
    "iud_16_23w",
    "iud_24_36w",
    "iud_gte_37w",
    "crl_1",
    "nt1",
    "bpd3",
    "us_gestation_weeks1",
    "days1",
    "1st_trim_msb_date",
    "msb_manufacturer",
    "b_hcg",
    "_bhcg_MoM",
    "pappa",
    "pappa_MoM",
    "msb_plgf_pgml",
    "msb_plgf_MoM",
    "r_uterine_a_pi",
    "r_uterine_a_ri",
    "l_uterine_a_pi",
    "l_uterine_a_ri"
]

SCAN_FIELDS = [
    "date_of_exam",
    "maternal_age_at_exam",
    "crl_1",
    "nt1",
    "bpd3",
    "us_gestation_weeks1",
    "days1",
    "1st_trim_msb_date",
    "msb_manufacturer",
    "b_hcg",
    "_bhcg_MoM",
    "pappa",
    "pappa_MoM",
    "msb_plgf_pgml",
    "msb_plgf_MoM",
    "r_uterine_a_pi",
    "r_uterine_a_ri",
    "l_uterine_a_pi",
    "l_uterine_a_ri"
]

OUTCOME_FIELDS = [

]

DROP_FIELDS = [
    "gest1"
]


def convert_field_name(name):
    tidy_name = name.lower()\
                    .replace(" < ", "_lt_") \
                    .replace(" >=", "_gte_") \
                    .replace(")", "") \
                    .replace("(", "") \
                    .replace("msb:", "msb") \
                    .replace(" ", "_") \
                    .replace(":", "_") \
                    .replace("-", "_") \
                    .replace("/", "") \
                    .replace(".", "_")
    tidy_name = sub("^_", "", tidy_name)
    tidy_name = sub("^1st", "first", tidy_name)
    return tidy_name


def generate_field_map_template(scan=None, df=None):
    """
    Creates a template field map, make sure not to overwrite ones that have been edited
    This makes it easier to convert different files into the same format with the same fields
    and types
    :param df:
    :return:
    """
    converted = sorted(map(convert_field_name, df.keys()))
    mapping = dict(
        baby=[],
        parent=converted,
        scan=[],
        drop=[],
        rename=[{"from": "a field", "to": "a new field"}]
    )

    return mapping


def save_mapping_template(mapping=None, template_root=None, filename=None):
    mapping_root = join(settings.DATA_ROOT, template_root)

    if not exists(mapping_root):
        makedirs(mapping_root)

    out_fname = join(mapping_root, "{0}.json".format(splitext(filename)[0]))

    with open(out_fname, "wb") as f:
        json.dump(mapping, f, indent=4)


def tidy_field_names(df=None):
    df.rename(columns=dict(zip(df.keys(), map(convert_field_name, df.keys()))), inplace=True)
    return df


def convert_file_type(center_root=None, filename=None):
    """
    To get started, need to get a template for the normalised field names
    :param center_root:
    :param filename:
    :return:
    """
    orig_fname = join(settings.DATA_ROOT, center_root, filename)
    cache_fname = join(settings.DATA_ROOT, center_root, "{0}.p".format(splitext(filename)[0]))

    if not exists(orig_fname):
        raise Exception("File not found: {0}".format(orig_fname))

    if exists(cache_fname):
        print "Loading cached {0}".format(cache_fname)
        with open(cache_fname, "rb") as f:
            orig = pickle.load(f)
    else:
        orig = xlxs2dataframe(orig_fname, cache_fname)

    print "Have {0} rows and {1} cols".format(len(orig), len(orig.keys()))

    return orig


def xlxs2dataframe(orig_fname=None, cache_fname=None):
    print "Loading {0}".format(orig_fname)

    orig = pd.read_excel(orig_fname)

    print "Found {0} lines".format(len(orig))

    print "Saving to {0}".format(cache_fname)
    with open(cache_fname, "wb") as f:
        pickle.dump(orig, f)

    return orig


if __name__ == "__main__":
    """
    Main function just here to make things easier when testing etc
    """

    files = dict(
        Centre1=[("scan1", "PREST1_v2.xlsx", ),
                 ("scan2", "PREST2_v2.xlsx", ),
                 ("scan3", "PREST3_v2.xlsx", )],
        # Centre2=[("scan1", "PREST1_v2 Centre2.xlsx", ),
        #          ("scan2", "PREST2_v2 Centre2.xlsx", ),
        #          ("scan3", "PREST3_v2 Centre2.xlsx", )]
    )

    # files = dict(
    #     Centre1=[("scan1", "PREST1_v2.xlsx", )
    #              #, ("scan2", "PREST2_v2.xlsx", ),
    #              #, ("scan3", "PREST3_v2.xlsx", )
    #     ]
    # )

    for center, files in files.iteritems():
        cfiles = OrderedDict()
        for scan, file in files:
            df = convert_file_type(center, file)
            df = tidy_field_names(df)
            cfiles[scan] = df
            # mapping = generate_field_map_template(scan, df)
            # save_mapping_template(mapping, "field_templates", "{0}.{1}".format(center, file))
        join_center_scans(cfiles)