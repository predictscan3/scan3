from collections import defaultdict
import pandas as pd
import numpy as np

from scan3 import settings


def join_center_scans(dfs_by_scan=None, sources=None):
    """
    Given the raw data by scan for a single center, join them together keyed on baby_id (which will be generated)
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
            filter_dates = list(filter(lambda x: x >= scan_date, dates))

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
        return scan_df2, missing_patient_ids

    if True:
        print("Process scan2")
        scan2, scan2_missing_patients = process_subsequent_scan(dfs_by_scan["scan2"], "scan2")
        missing_patient_info["scan2"] = scan2_missing_patients

    if True:
        print("Process scan3")
        scan3, scan3_missing_patients = process_subsequent_scan(dfs_by_scan["scan3"], "scan3")
        missing_patient_info["scan3"] = scan3_missing_patients

    # Then, for each baby, we can separate out the different data:
    # - parent/baby meta
    # - scan
    # - outcome

    # TODO Do a proper check using sets to identify missing patients, especially when we have ones in scan2 and 3 but
    # not scan 1

    # In the end we want a big stacked table, with one row per scan
    # The id is therefore the baby_id + scan_date
    # The parent data and other meta data will be present in each row, the tidying up of this data needs to happen
    # before we do the denormalisation, otherwise record counts will be inaccurate
    # At some point also need to convert dates etc, so that we can do better date logic and analysis

    # First of all, just identify the scan fields and stack those together

    # There may be multiple scans per baby in each dataset, but don't need to worry about that as when we (later) do
    # some filtering based on scan date, these will be taken care of
    # Might be useful to have a report with the number of scans per baby though, as a sanity check if nothing else

    # First of all, sort out the indexes, they need to be baby_id + scan date
    def drop_bad_rows(name=None, df=None):
        """
        There are lots of duplicated rows, but only one has actual scan data, they both have other data though
        Try and identify and drop these useless rows, then check if we still have duplicates before reindexing.
        Note that we have to split and stack before we can reindex, as we'll be joining on the indexes.
        """

        # Identify rows without any scan data
        pure_scan_fields = list(set(settings.SCAN_FIELDS) - {"date_of_exam"})
        df["missing_scan_fields"] = 0

        for f in pure_scan_fields:
            df["missing_scan_fields"] += df[f].map(lambda x: 1 if x is None or np.isnan(x) else 0)

        bad = (df.missing_scan_fields == (len(pure_scan_fields) - 1))

        print("Dropping {0} rows from {1} as they have no scan fields".format(len(bad[bad == True]), name))

        df.drop(df.index[bad == True], axis=0, inplace=True)

        # Start setting up the new index, and check for duplicates again
        df["idx"] = df["baby_id"] + "_" + df["date_of_exam"]
        dupes = df.groupby("idx").size()
        dupes = dupes[dupes > 1]

        print("Dropping {0} rows from {1} as they have duplicate scan ids".format(len(dupes), name))
        df_new = df.drop_duplicates(subset=["idx"], keep="first")

        return df_new

    def reindex(df=None):
        df_new = df.set_index(df.idx, verify_integrity=True)
        return df_new

    scan1_fields = settings.PARENT_FIELDS + ["baby_id"]
    scan2_fields = settings.SCAN_FIELDS + ["baby_id"]
    scan3_fields = settings.OUTCOME_FIELDS + ["baby_id"]

    print("Reindexing scans")
    scan1_new = drop_bad_rows("scan1", scan1)
    scan2_new = drop_bad_rows("scan2", scan2)
    scan3_new = drop_bad_rows("scan3", scan3)

    print("Updating scans with parent and outcome data")

    # Need to supplement each scan file with the fields it needs from the others, before reindexing, as after that the
    # index contains the scan date, which is different for each scan.
    # So we need a separate scan1 and scan3 df with just the outcome fields, with the baby_id as the index so we can
    # use it to supplement the others.

    scan1_minimal = scan1_new.drop_duplicates(subset=["baby_id"], keep="first")
    scan1_minimal.set_index(scan1_minimal.baby_id, inplace=True)

    scan3_minimal = scan3_new.drop_duplicates(subset=["baby_id"], keep="first")
    scan3_minimal.set_index(scan3_minimal.baby_id, inplace=True)

    # Update scan 2 and 3 with the parent data from scan 1
    scan2_supp = scan2_new[scan2_fields + ["idx"] + settings.DEBUG_FIELDS].join(scan1_minimal[scan1_fields],
                                                                                on="baby_id",
                                                                                how="inner",
                                                                                lsuffix="_xs2",
                                                                                rsuffix="_xs1")
    scan3_supp = scan3_new[scan3_fields + ["idx"] + settings.DEBUG_FIELDS].join(scan1_minimal[scan1_fields],
                                                                                on="baby_id",
                                                                                how="inner",
                                                                                lsuffix="_xs3",
                                                                                rsuffix="_xs1")

    # Update scan 1 and 2 with the outcome data from scan 3
    scan1_supp = scan1_new[scan1_fields + ["idx"] + settings.DEBUG_FIELDS].join(scan3_minimal[scan3_fields],
                                                                                on="baby_id",
                                                                                how="inner",
                                                                                lsuffix="_xs1",
                                                                                rsuffix="_xs3")
    scan2_supp = scan2_supp.join(scan3_minimal[scan3_fields],
                                 on="baby_id",
                                 how="inner",
                                 lsuffix="_xs2",
                                 rsuffix="_xs3")

    print("Scan1 final size: {0}".format(len(scan1_supp)))
    print("Scan2 final size: {0}".format(len(scan2_supp)))
    print("Scan3 final size: {0}".format(len(scan3_supp)))

    # Reindex so that we can properly stack everything
    scan1_final = reindex(scan1_supp)
    scan2_final = reindex(scan2_supp)
    scan3_final = reindex(scan3_supp)

    # Drop any extra cols
    for scan in [scan1_final, scan2_final, scan3_final]:
        drops = []
        for suffix in ["_xs1", "_xs2", "_xs3"]:
            for k in scan.keys():
                if k.endswith(suffix):
                    drops.append(k)
        if len(drops) > 0:
            scan.drop(axis=1, labels=drops, inplace=True)

    print("Join all scans together")

    # Stack everything together
    final = pd.concat([scan1_final, scan2_final, scan3_final])

    # TODO Generate some stats here and do some checks to make sure we dont have duplicate cols etc
    print("Final dataset has {0} rows".format(len(final)))

    return final