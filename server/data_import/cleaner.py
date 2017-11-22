import pandas as pd
import settings
from os.path import join, exists, splitext
from os import makedirs
import cPickle as pickle
import json


def generate_field_map_template(df=None):
    """
    Creates a template field map, make sure not to overwrite ones that have been edited
    This makes it easier to convert different files into the same format with the same fields
    and types
    :param df:
    :return:
    """
    mapping = {}

    for k in df.keys():
        mapping[k] = dict(
            target=k.lower().replace(" ", "_").replace("-", "_").replace(".", "_"),
            table="parent",  # example
            drop=False,
            dtype=df[k].dtype.name
        )

    return mapping


def save_mapping_template(mapping=None, template_root=None, filename=None):
    mapping_root = join(settings.DATA_ROOT, template_root)

    if not exists(mapping_root):
        makedirs(mapping_root)

    out_fname = join(mapping_root, "{0}.json".format(splitext(filename)[0]))

    with open(out_fname, "wb") as f:
        json.dump(mapping, f, indent=4)


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
    # files = dict(
    #     Centre1=["PREST1_v2.xlsx"]
    # )
    files = dict(
        Centre1=[("scan1", "PREST1_v2.xlsx", ),
                 ("scan2", "PREST2_v2.xlsx", ),
                 ("scan3", "PREST3_v2.xlsx", )]
    )

    for center, files in files.iteritems():
        for scan, file in files:
            df = convert_file_type(center, file)
            mapping = generate_field_map_template(df)
            save_mapping_template(mapping, "field_templates", "{0}.{1}".format(center, file))