from collections import defaultdict
import numpy as np
from datetime import datetime

from scan3 import settings


def join_center_scans(dfs_by_scan=None, sources=None):
    """
    Given the raw data by scan for a single center, join them together keyed on baby_id (which will be generated)
    The final file will have one row per baby, with scan fields by trimester
    TODO Refactor the ID generation functions
    :param dfs_by_scan:
    :param sources:
    :return:
    """
    missing_patient_info = {}

    # For each scan, generate a baby id for each row, noting that a parent may have multiple babies during the
    # data collection period
    # By keeping a record of the conception date of each baby for a parent, we can easily match the right baby id
    # from subsequent scans, as they will always be < the next conception date, and > the conception date of the
    # baby we're processing

    scan1 = dfs_by_scan["scan1"]

    # Base the baby ID on the estimated delivery date, use the ultrasound one if available otherwise fall back to
    # the LMP one
    # There must be a faster way than brute force looping

    # Some of the values are coming in as "nan" which means they're numbers which messes other things up
    print("Sort out missing EDD values")

    scan1.edd_us = scan1.edd_us.astype(str)
    scan1.edd_lmp = scan1.edd_lmp.astype(str)

    scan1["edd"] = [None] * len(scan1)
    good_edd_us = (scan1.edd_us.map(len) == 10)  # Date format is YYYY-MM-DD

    scan1.loc[good_edd_us, "edd"] = scan1.edd_us[good_edd_us]
    scan1.loc[good_edd_us == False, "edd"] = scan1.edd_lmp[good_edd_us == False]

    # TODO Drop fields where we still don't have an EDD, we can't process them

    scan1["baby_id"] = scan1["patients_id"].astype(str) + "_" + scan1["edd"]

    # Store EDD by parent, then for a given scan date, just find the min EDD that is > the scan date, to generate the
    # baby id
    print("Store EDD by parent, for baby ID generation")
    edd_by_parent = defaultdict(list)

    # Do this in stages rather than one by one brute force, otherwise it's really slow
    # Where we just have one baby per parent, simple map
    num_scans = scan1.groupby("patients_id").size()
    just_one_patients = num_scans[num_scans == 1].index
    just_one_edd = scan1[scan1.patients_id.isin(just_one_patients)]

    edd_by_parent.update(dict((p, [e]) for (p, e) in zip(just_one_edd.patients_id, just_one_edd.edd)))

    # Now process those with more than one baby/scan.  Quicker to do it this way.
    more_patients = num_scans[num_scans != 1].index

    for patients_id in more_patients:
        edd_by_parent[patients_id] = sorted(scan1.edd[scan1.patients_id == patients_id])

    def get_baby_id(patient_id=None, scan_date=None):
        dates = edd_by_parent.get(patient_id)

        if dates is None:
            return None

        if len(dates) == 1:
            return "{0}_{1}".format(patient_id, dates[0])
        else:
            # Filter out any dates less than the scan date, then take the min of what's left (because we already
            # sorted the dates
            # TODO Massive hack around dates, sort this out
            for v in (dates, [scan_date], ):
                if isinstance(v[0], str):
                    v = map(lambda x: datetime.strptime(x, "%Y-%m-%d"), v)
            try:
                if isinstance(scan_date, datetime):
                    sd = scan_date.strftime("%Y-%m-%d")
                else:
                    sd = scan_date
                filter_dates = list(filter(lambda x: x >= sd, dates))
            except Exception as e:
                print(e)

            if len(filter_dates) == 0:
                # Baby must be overdue, so take the most recent EDD
                # The estimated delivery date could be before the scan date, if the baby is overdue
                # if this happens, then I think we just take the most recent estimated delivery date,
                # it must be the right one?
                baby_edd = dates[-1]
            else:
                baby_edd = filter_dates[0]

            return "{0}_{1}".format(patient_id, baby_edd)

    # Now can process the next scan, need to supplement it with baby ids before we can join the data sets
    def process_subsequent_scan(scan_df=None, name=None):
        baby_ids = []
        missing_patient_ids = []
        for row in scan_df.iterrows():
            baby_id = get_baby_id(row[1].patients_id, row[1].date_of_exam)
            if baby_id is None:
                missing_patient_ids.append(row[1].patients_id)
            baby_ids.append(baby_id)

        scan_df["baby_id"] = baby_ids

        print("Found {0} {1} patients with no match in scan1".format(len(missing_patient_ids), name))

        # Need to drop those rows with missing baby ids, can't do anything with them
        missing = scan_df.baby_id.isnull()
        print("Dropping {0} rows from {1}".format(len(missing[missing == True]), name))
        scan_df2 = scan_df.drop(scan_df[missing].index)

        # Turn the scan date field into a date, makes things more explicit?
        # Also the EDD column
        scan_df2["date_of_exam"] = scan_df.date_of_exam.map(lambda x: x if isinstance(x, datetime) else datetime.strptime(x, "%Y-%m-%d"))
        # scan_df2["edd"] = scan_df.edd.map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        return scan_df2, missing_patient_ids

    if True:
        print("Find baby id for scan2")
        scan2, scan2_missing_patients = process_subsequent_scan(dfs_by_scan["scan2"], "scan2")
        missing_patient_info["scan2"] = scan2_missing_patients

    if True:
        print("Find baby id for scan3")
        # TODO Check that we don't drop rows where there is no scan3 info, as that would imply the baby was fine
        scan3, scan3_missing_patients = process_subsequent_scan(dfs_by_scan["scan3"], "scan3")
        missing_patient_info["scan3"] = scan3_missing_patients

    # Then, for each baby, we can separate out the different data:
    # - parent/baby meta
    # - scan
    # - outcome

    # TODO Do a proper check using sets to identify missing patients, especially when we have ones in scan2 and 3 but
    # not scan 1

    # There may be multiple scans per baby in each dataset, take the last one for now, need to do some work to make
    # sure that we collapse data where possible, as there might be two scans and the data is only all present when
    # combining them
    # Also, for now just take the last scan date if there are duplicates.  Need to do some work to pick up scan2
    # if the baby is dead in scan3

    def drop_bad_rows(name=None, df=None):
        """
        There are lots of duplicated rows, but only one has actual scan data, they both have other data though
        Try and identify and drop these useless rows, then check if we still have duplicates before reindexing.
        Note that we have to split and stack before we can reindex, as we'll be joining on the indexes.
        """

        # Identify rows without any scan data
        pure_scan_fields = list(set(settings.FIELDS_BY_SCAN_FILE[name]) - {"date_of_exam"})
        df["missing_scan_fields"] = 0

        def is_bad(x):
            if x is None:
                return 1
            elif isinstance(x, str):
                return 1 if len(x) == 0 else 0
            else:
                return 1 if np.isnan(x) else 0

        for f in set(pure_scan_fields).intersection(set(df.keys())):
            # TODO This needs tidying up, bit slow to check the type....
            print("\tChecking {0}.{1}".format(name, f))
            df["missing_scan_fields"] += df[f].map(is_bad)

        bad = (df.missing_scan_fields == (len(pure_scan_fields) - 1))

        print("Dropping {0} rows from {1} as they have no scan fields".format(len(bad[bad == True]), name))

        df.drop(df.index[bad == True], axis=0, inplace=True)

        # Start setting up the new index, and check for duplicates again
        df["idx"] = df["baby_id"]
        dupes = df.groupby("idx").size()
        dupes = dupes[dupes > 1]

        print("Dropping {0} rows from {1} as they have duplicate scan ids".format(len(dupes), name))
        # TODO Need to sort by the scan date here, so that the duplicate dropping works properly
        df_new = df.drop_duplicates(subset=["idx"], keep="last")

        return df_new

    def rename_scan_fields(name=None, df=None):
        scan_fields = settings.FIELDS_BY_SCAN_FILE[name]
        rename = {}
        for f in scan_fields:
            # TODO Need to actually use the trimester field that is generated from the scan date - conception date
            rename[f] = "t{0}_{1}".format(name[-1], f)
        return df.rename(columns=rename)

    def rename_parent_fields(name=None, df=None):
        p_fields = settings.PARENT_FIELDS
        rename = {}
        for f in p_fields:
            rename[f] = "dem_{0}".format(f)
        return df.rename(columns=rename)

    def rename_debug_fields(df=None):
        p_fields = settings.FINAL_DEBUG_FIELDS
        rename = {}
        for f in p_fields:
            rename[f] = "debug_{0}".format(f)
        return df.rename(columns=rename)

    def reindex(df=None):
        df_new = df.set_index(df.idx, verify_integrity=True)
        df_new.drop(axis=1, labels=["idx"], inplace=True)
        return df_new

    print("Reindexing scans")
    scan1_new = drop_bad_rows("scan1", scan1)
    scan2_new = drop_bad_rows("scan2", scan2)
    scan3_new = drop_bad_rows("scan3", scan3)

    print("Renaming scan fields")
    scan1_rename = rename_parent_fields("scan1", rename_scan_fields("scan1", scan1_new))
    scan2_rename = rename_parent_fields("scan2", rename_scan_fields("scan2", scan2_new))
    scan3_rename = rename_parent_fields("scan3", rename_scan_fields("scan3", scan3_new))

    print("Joining scans")

    # First need to set the index to baby_id
    scan1_rename = reindex(scan1_rename)
    scan2_rename = reindex(scan2_rename)
    scan3_rename = reindex(scan3_rename)

    # Join together
    scan1_2 = scan1_rename.join(scan2_rename, how="inner", lsuffix="_xs2", rsuffix="_xs1")
    # TODO Need to make sure we don't lose rows that don't have scan3 info
    scan1_2_3 = scan1_2.join(scan3_rename, how="inner", rsuffix="_xs3")

    print("Scan final size: {0}".format(len(scan1_2_3)))

    # Drop any extra cols
    for scan in [scan1_2_3]:
        drops = []
        for suffix in ["_xs1", "_xs2", "_xs3"]:
            for k in scan.keys():
                if k.endswith(suffix):
                    drops.append(k)
        if len(drops) > 0:
            scan.drop(axis=1, labels=drops, inplace=True)

    # Drop/Rename final cols
    scan1_2_3 = rename_debug_fields(scan1_2_3)
    scan1_2_3.drop(axis=1, labels=settings.FINAL_DROP_FIELDS, inplace=True)

    return scan1_2_3
