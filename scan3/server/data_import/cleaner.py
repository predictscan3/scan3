import pandas as pd
from scan3 import settings
from os.path import join, exists, splitext, dirname
from os import makedirs
try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
from collections import OrderedDict
from re import sub

# from scan3.server.data_import.joiner import join_center_scans
from scan3.server.data_import.joiner_by_baby import join_center_scans
from scan3.server.data_import.enrich import add_calcd_fields, apply_filters


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
                    .replace(".", "_") \
                    .replace("mat_age_at_exam", "maternal_age_at_exam") \
                    .replace("patient_id", "patients_id")
    tidy_name = sub("^_", "", tidy_name)
    tidy_name = sub("^1st", "first", tidy_name)

    # Process any synonyms
    tidy_name = settings.FIELD_SYNOMYMS.get(tidy_name, tidy_name)

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


def drop_cols(scan=None, df=None):
    drops = list(set(settings.DROP_FIELDS_BY_SCAN[scan]).intersection(set(df.keys())))
    print("Dropping fields from {0}: {1}".format(scan, ", ".join(drops)))
    df.drop(axis=1, labels=drops, inplace=True)
    return df


def convert_file_type(center_root=None, filename=None, force=False):
    """
    To get started, need to get a template for the normalised field names
    :param center_root:
    :param filename:
    :return:
    """
    orig_fname = join(settings.DATA_IN_ROOT, center_root, filename)
    cache_fname = join(settings.DATA_OUT_ROOT, center_root, "{0}.p".format(splitext(filename)[0]))

    if not exists(orig_fname):
        raise Exception("File not found: {0}".format(orig_fname))

    if exists(cache_fname) and force is False:
        print("Loading cached {0}".format(cache_fname))
        with open(cache_fname, "rb") as f:
            orig = pickle.load(f)
    else:
        orig = xlxs2dataframe(orig_fname, cache_fname)

    # Store some debug info
    orig["center"] = center_root
    orig["filename"] = filename

    print("Have {0} rows and {1} cols".format(len(orig), len(orig.keys())))

    return orig


def xlxs2dataframe(orig_fname=None, cache_fname=None):
    print("Loading {0}".format(orig_fname))

    orig = pd.read_excel(orig_fname)

    print("Found {0} lines".format(len(orig)))

    if not exists(dirname(cache_fname)):
        makedirs(dirname(cache_fname))
    print("Saving to {0}".format(cache_fname))
    with open(cache_fname, "wb") as f:
        pickle.dump(orig, f)

    return orig


def inspect_scan_fields(cfiles=None):
    """
    Useful for figuring out where common fields are and spotting duplicates
    :param cfiles:
    :return:
    """
    # TODO Do some more work here, there are some scan fields in 2 and 3 but not one, need to ask Basky
    scan1_cols = set(cfiles["scan1"].columns)
    scan2_cols = set(cfiles["scan2"].columns)
    scan3_cols = set(cfiles["scan3"].columns)

    print("Scan 1+2+3 intersect == scan fields")
    print("SCAN_FIELDS = " + str(sorted(list(scan1_cols.intersection(scan2_cols).intersection(scan3_cols)))))

    print("Scan 1-2-3 == parent info")
    print("PARENT_FIELDS = " + str(sorted(list(scan1_cols - scan2_cols - scan3_cols))))

    print("Scan 3-2-1 == outcome info")
    print("OUTCOME_FIELDS = " + str(sorted(list(scan3_cols - scan2_cols - scan1_cols))))

    print("Scan 2-1-3 == extra scan2 fields")
    print(scan2_cols - scan1_cols - scan3_cols)


def clean_and_join(centers=None, fname_p=None, fname_csv=None):

    files = dict(
        Centre1=[("scan1", "PREST1_v2.xlsx", ),
                 ("scan2", "PREST2_v2.xlsx", ),
                 ("scan3", "PREST3_v2.xlsx", )],
        Centre2=[("scan1", "PREST1_v2 Centre2.xlsx", ),
                 ("scan2", "PREST2_v2 Centre2.xlsx", ),
                 ("scan3", "PREST3_v2 Centre2.xlsx", )]
    )

    # files = dict(
    #     Centre1=[("scan1", "PREST1_v2.xlsx", )
    #              #, ("scan2", "PREST2_v2.xlsx", ),
    #              #, ("scan3", "PREST3_v2.xlsx", )
    #     ]
    # )

    force = False

    # Load all files and tidy up the field names
    tidy_files = OrderedDict()
    for center, sfiles in iter(files.items()):
        if center in centers:
            cfiles = OrderedDict()
            for scan, file in sfiles:
                df = convert_file_type(center, file, force=force)
                df = tidy_field_names(df)
                df = drop_cols(scan, df)
                cfiles[scan] = df
            tidy_files[center] = cfiles

    # Check whether we have any important fields missing, at the moment center1 is our default
    # base_center = "Centre1"
    # base = tidy_files[base_center]
    # for center in (set(tidy_files.keys()) - set(base_center)):
    #     cfiles = tidy_files[center]
    #     for scan, df in iter(cfiles.items()):
    #         missing = set(base[scan].keys()) - set(df.keys())
    #         extra = set(df.keys()) - set(base[scan].keys())
    #         print("Extra in {0}, {1}:".format(center, scan))
    #         print("\n".join(sorted(extra)))
    #         print()
    #         print("Missing from {0}, {1}:".format(center, scan))
    #         print("\n".join(sorted(missing)))
    #         print()

    # Join the center files into one, indexed on baby_id (composite of parent_id and EDD)
    center_dfs = []
    for center, cfiles in iter(tidy_files.items()):
        center_df = join_center_scans(center, cfiles, files)
        center_dfs.append(center_df)

        c_fname_p = join(outroot, "{0}_by_baby.p".format(center))
        c_fname_csv = join(outroot, "{0}_by_baby.csv".format(center))

        print("Writing {0} rows for {1} to {2}".format(len(center_df), center, c_fname_csv))
        center_df.to_pickle(c_fname_p)
        center_df.to_csv(c_fname_csv, index_label="idx")

    # Concatenate everything into one big file and save
    final = pd.concat(center_dfs)

    print("Writing {0} rows for {1} to {2}".format(len(final), ", ".join(tidy_files.keys()), fname_p))
    final.to_pickle(fname_p)
    final.to_csv(fname_csv, index_label="idx")

    return final


def enrich(df=None):
    print("Enriching data")
    df = add_calcd_fields(df)
    df = apply_filters(df)
    return df


if __name__ == "__main__":
    """
    Main function just here to make things easier when testing etc
    """
    print("Project root is: {0}".format(settings.PROJECT_ROOT))
    print("Data source is: {0}".format(settings.DATA_IN_ROOT))
    print("Data target is: {0}".format(settings.DATA_OUT_ROOT))

    CENTER_FILTER = {"Centre1", "Centre2"}  # to help with debugging
    # CENTER_FILTER = {"Centre1"}

    JOIN = True
    ENRICH = True

    outroot = join(settings.DATA_OUT_ROOT, "data_staging")
    if not exists(outroot):
        makedirs(outroot)

    joined_fname_p = join(outroot, "all_by_baby_v{0}.p".format(settings.JOINER_VERSION))
    joined_fname_csv = join(outroot, "all_by_baby_v{0}.csv".format(settings.JOINER_VERSION))

    enriched_fname_p = join(outroot, "all_by_baby_enriched_v{0}.p".format(settings.JOINER_VERSION))
    enriched_fname_csv = join(outroot, "all_by_baby_enriched_v{0}.csv".format(settings.JOINER_VERSION))

    if JOIN:
        joined = clean_and_join(CENTER_FILTER, joined_fname_p, joined_fname_csv)
    else:
        joined = pd.read_pickle(joined_fname_p)

    if ENRICH:
        enriched = enrich(joined)
        print("Writing enriched data to {0}".format(enriched_fname_csv))
        enriched.to_csv(enriched_fname_csv, index_label="idx")
        enriched.to_pickle(enriched_fname_p)