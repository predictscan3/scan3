from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from os.path import exists, join
from os import makedirs

from scan3 import settings
from scan3.server.data_import.enrich import convert_dates, to_datetime, add_scan_calcd_fields


def dump_debug_df(name=None, df=None):
    outroot = join(settings.DATA_OUT_ROOT, "data_debug")
    if not exists(outroot):
        makedirs(outroot)
    df.to_csv(join(outroot, "{0}.csv".format(name)))


def join_center_scans(center=None, dfs_by_scan=None, sources=None):
    """
    Given the raw data by scan for a single center, join them together keyed on baby_id (which will be generated)
    The final file will have one row per baby, with scan fields by trimester
    TODO Refactor the ID generation functions
    :param center:
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

    scan1 = add_scan_calcd_fields("t1", scan1)

    # TODO Drop fields where we still don't have an EDD, we can't process them

    scan1["baby_id"] = scan1["patients_id"].astype(str) + "_" + scan1["edd"]

    # Conver the EDD to a date to make things easier
    scan1["edd"] = to_datetime(scan1.edd)

    # Store the mothers age by scan1 date, so that we can verify it when matching scans
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
        edd_by_parent[patients_id] = sorted(set(scan1.edd[scan1.patients_id == patients_id]))

    def get_baby_id(patient_id=None, scan_date=None, record=None):
        edd_dates = edd_by_parent.get(patient_id)

        if edd_dates is None:
            return None

        if len(edd_dates) == 1:
            if edd_dates[0] < scan_date:
                # Is the baby overdue or is this a different baby with missing scan info?
                if (scan_date - edd_dates[0]).days < 21:  # 3 weeks, just a guess
                    return "{0}_{1:%Y-%m-%d}".format(patient_id, edd_dates[0])
                else:
                    return None
            else:
                return "{0}_{1:%Y-%m-%d}".format(patient_id, edd_dates[0])
        else:
            # Filter out any dates less than the scan date, then take the min of what's left (because we already
            # sorted the dates)
            filter_dates = list(filter(lambda x: x >= scan_date, edd_dates))

            if len(filter_dates) == 0:
                # Baby must be overdue, so take the most recent EDD
                # The estimated delivery date could be before the scan date, if the baby is overdue
                # if this happens, then I think we just take the most recent estimated delivery date,
                # it must be the right one?
                if (scan_date - edd_dates[0]).days < 21:  # 3 weeks, just a guess
                    baby_edd = edd_dates[0]
                else:
                    return None
            else:
                baby_edd = filter_dates[0]

            return "{0}_{1:%Y-%m-%d}".format(patient_id, baby_edd)

    # Now can process the next scan, need to supplement it with baby ids before we can join the data sets
    def process_subsequent_scan(scan_df=None, name=None):
        baby_ids = []
        missing_patient_ids = []
        for row in scan_df.iterrows():
            baby_id = get_baby_id(row[1].patients_id, row[1].date_of_exam, row[1])
            if baby_id is None:
                missing_patient_ids.append(row[1].patients_id)
                # Generate a baby id, so that this data can be used, even if we lack demographic data
                # Do this by working out a likely EDD from the scan date and gestational age
                ecd = row[1].date_of_exam - timedelta(weeks=row[1].us_gestation_weeks1, days=row[1].days1)
                edd = ecd + timedelta(weeks=40)
                baby_id = "{0}_{1:%Y-%m-%d}_G".format(row[1].patients_id, edd)
            baby_ids.append(baby_id)

        scan_df["baby_id"] = baby_ids

        print("Found {0} {1} patients with no match in scan1".format(len(missing_patient_ids), name))

        return scan_df, missing_patient_ids

    if True:
        print("Find baby id for scan2")
        scan2 = add_scan_calcd_fields("t2", dfs_by_scan["scan2"])
        scan2, scan2_missing_patients = process_subsequent_scan(scan2, center + ".scan2")
        missing_patient_info["scan2"] = scan2_missing_patients

    if True:
        print("Find baby id for scan3")
        scan3 = add_scan_calcd_fields("t3", dfs_by_scan["scan3"])
        # TODO Check that we don't drop rows where there is no scan3 info, as that would imply the baby was fine
        scan3, scan3_missing_patients = process_subsequent_scan(scan3, center + ".scan3")
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

    def drop_bad_rows(center=None, scan=None, df=None):
        """
        There are lots of duplicated rows, but only one has actual scan data, they both have other data though
        Try and identify and drop these useless rows, then check if we still have duplicates before reindexing.
        Note that we have to split and stack before we can reindex, as we'll be joining on the indexes.
        """

        # Identify rows without any scan data
        name = "{0}.{1}".format(center, scan)
        pure_scan_fields = list(set(settings.CHECK_FIELDS_BY_SCAN_FILE[scan]) - {"date_of_exam"})
        df["missing_scan_fields"] = 0

        def is_bad(x):
            if x is None:
                return 1
            elif isinstance(x, str):
                return 1 if len(x) == 0 else 0
            elif isinstance(x, datetime):
                return 1 if np.isnat(x) else 0
            else:
                return 1 if np.isnan(x) else 0

        for f in set(pure_scan_fields).intersection(set(df.keys())):
            try:
                # TODO This needs tidying up, bit slow to check the type....
                print("\tChecking {0}.{1}.{2}".format(center, scan, f))
                df["missing_scan_fields"] += df[f].map(is_bad)
            except Exception as e:
                print(str(e))

        # Find duplicate scans, then drop any of them that have no scan fields
        print("Determining which {0} records to drop, keeping the one with the most scan data".format(name))
        bad_idx = []
        for baby_id, dupes in df.groupby("baby_id"):
            if len(dupes) > 1:
                # Keep the one with the most scan fields
                keep_dupes = dupes[dupes.missing_scan_fields == min(dupes.missing_scan_fields)]

                if len(keep_dupes) > 1:
                    # More than one eligible, so just take the most recent one
                    keep_dupe_idx = keep_dupes[keep_dupes.date_of_exam == max(keep_dupes.date_of_exam)]

                    if len(keep_dupe_idx) > 1:
                        # Multiple scans on the same day, with the same number of scan fields present, unlikely
                        # Spot checking a few shows that they often have different fields present, so we should
                        # merge the two
                        # TODO Check with Basky about merging rows, can we take the average when a value is the same
                        # in both or should be just take the last one?
                        # print("Multiple {0} for {1} on {2:%d %b %Y} ({3})".format(name,
                        #                                                           baby_id,
                        #                                                           keep_dupe_idx.iloc[0].date_of_exam,
                        #                                                           len(keep_dupe_idx)))
                        pass
                    keep_dupe_idx = keep_dupe_idx.index[0]
                else:
                    keep_dupe_idx = keep_dupes.index[0]

                drop_dupe_idx = set(dupes.index)
                drop_dupe_idx.remove(keep_dupe_idx)
                bad_idx += list(drop_dupe_idx)

        print("Dropping {0} duplicate rows from {1} (keeping records with most scan data)".format(len(bad_idx), name))
        dump_debug_df("{0}_dupes_least_scan_fields".format(name), df[df.index.isin(bad_idx)])

        df_no_dupes = df.drop(bad_idx, axis=0)

        # Start setting up the new index, and check for duplicates again (there shouldn't be any)
        df_no_dupes["idx"] = df_no_dupes["baby_id"]

        # Sort by the scan date so that we just keep the most recent one
        # TODO This needs a bit of extra logic for the third trimester scan (if the baby is dead in the last scan
        # TODO then we need the one before)
        df_no_dupes.sort_values(by=["baby_id", "date_of_exam"], inplace=True)
        df_new = df_no_dupes.drop_duplicates(subset=["idx"], keep="last")

        if len(df_new) < len(df_no_dupes):
            print("Dropped {0} more rows from {1} as they were duplicated".format(len(df_no_dupes) - len(df_new), name))
            dump_debug_df("{0}_dupe_baby_id".format(name), df_no_dupes[df_no_dupes.index.isin(df_new.index) == False])

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
    scan1_new = drop_bad_rows(center, "scan1", scan1)
    scan2_new = drop_bad_rows(center, "scan2", scan2)
    scan3_new = drop_bad_rows(center, "scan3", scan3)

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
    scan1_2 = scan1_rename.join(scan2_rename, how="outer", lsuffix="_xs1", rsuffix="_xs2")
    joined = scan1_2.join(scan3_rename, how="outer", rsuffix="_xs3")

    # Sort out common fields
    joined["dem_center"] = center

    # Because we've done an outer join, need to make sure common fields are properly dealt with, eg dob
    def filter_ends(endswith, cols):
        return list(filter(lambda x: x.endswith(endswith), cols))

    def filter_starts(startswith, cols):
        return list(filter(lambda x: x.startswith(startswith), cols))

    def filter_contains(contains, cols):
        return list(filter(lambda x: x.find(contains) != -1, cols))

    def unique_date(x):
        uniq = np.unique(list(filter(lambda x: np.isnat(x) == False, x.values)))
        return uniq[0] if len(uniq) > 0 else None

    def unique_num(x):
        uniq = np.unique(list(filter(lambda x: np.isnan(x) == False, x.values)))
        return uniq[0] if len(uniq) > 0 else None

    joined["dem_dob"] = joined[filter_starts("dem_dob", joined.keys())].apply(unique_date, axis=1)
    joined["patients_id"] = joined[filter_starts("patients_id", joined.keys())].apply(unique_num, axis=1)

    # We need one maternal age for the model, doesn't matter too much which one
    joined["dem_mat_age"] = joined[filter_contains("mat_age_at_exam", joined.keys())].apply(unique_num, axis=1)

    print("{0} final size: {1}".format(center, len(joined)))

    # Drop any extra cols
    for scan in [joined]:
        drops = []
        for suffix in ["_xs1", "_xs2", "_xs3"]:
            for k in scan.keys():
                if k.endswith(suffix):
                    drops.append(k)
        if len(drops) > 0:
            scan.drop(axis=1, labels=drops, errors="ignore", inplace=True)

    # Drop/Rename final cols
    joined = rename_debug_fields(joined)
    joined.drop(axis=1, labels=settings.FINAL_DROP_FIELDS, errors="ignore", inplace=True)

    # Drop rows where the data appears to be incorrect
    # Convert dates
    joined = convert_dates(joined)

    bad_scan_dates = joined[(joined.t2_date_of_exam < joined.t1_date_of_exam) |
                            (joined.t3_date_of_exam < joined.t1_date_of_exam)]
    # TODO Actually, I think some of these are just where there is a missing scan, and my joining logic doesn't work
    # TODO properly.

    print("Found {0} rows where the scan dates must be wrong, eg scan3 < scan1, or scan2 < scan1".format(len(bad_scan_dates)))
    dump_debug_df("{0}_bad_scan_dates".format(center), bad_scan_dates)

    final = joined.drop(bad_scan_dates.index, axis=0)

    return final
