from collections import defaultdict
import numpy as np


def join_center_scans(dfs_by_scan=None):
    """
    Given the raw data by scan for a single center, join them together keyed on baby_id (which will be generated)
    :param dfs_by_scan:
    :return:
    """

    # For each scan, generate a baby id for each row, noting that a parent may have multiple babies during the
    # data collection period
    # By keeping a record of the conception date of each baby for a parent, we can easily match the right baby id
    # from subsequent scans, as they will always be < the next conception date, and > the conception date of the
    # baby we're processing

    scan = dfs_by_scan["scan1"]
    #print("\n".join(map(str, set(scan["edd_lmp"]))))

    # Base the baby ID on the estimated delivery date, use the ultrasound one if available otherwise fall back to
    # the LMP one
    # There must be a faster way than brute force looping

    # Some of the values are coming in as "nan" which means they're numbers which messes other things up
    print("Sort out missing EDD values")

    scan.edd_us = scan.edd_us.astype(str)
    scan.edd_lmp = scan.edd_lmp.astype(str)

    scan["edd"] = [None] * len(scan)
    good_edd_us = (scan.edd_us.map(len) == 10)

    scan.loc[good_edd_us, "edd"] = scan.edd_us[good_edd_us]
    scan.loc[good_edd_us == False, "edd"] = scan.edd_lmp[good_edd_us == False]

    # TODO Drop fields where we still don't have an EDD, we can't process them

    # for row in scan.iterrows():
    #     if isinstance(row[1].edd_us, str):
    #         edd.append(row[1].edd_us)
    #     elif isinstance(row[1].edd_lmp, str):
    #         edd.append(row[1].edd_lmp)
    #     else:
    #         print("Drop row {0}".format(row[0]))

    scan["baby_id"] = scan["patients_id"].astype(str) + "_" + scan["edd"]

    # Need a lookup of scan date to baby id, so we can work our way back to the baby id from a parent and scan date
    # Might need to do this within the parent grouping, need multiple levels anyway
    # baby_by_scan = dict()
    # for baby_id, data in scan.groupby("baby_id"):
    #     if len(data) > 1:
    #         raise Exception("Found {0} babies for scan {1}".format(len(data), baby_id))
    #     baby_by_scan[baby_id].app = baby
    #
    # # Get a lookup of scan dates by mother, so we can cope with the subsequent scans where a mother has multiple babies
    # scan_by_parent = defaultdict(list)
    # for parent, data in scan.groupby("patients_id"):
    #     scan_by_parent[parent] = sorted(set(data.date_of_exam))

    # LATEST
    # Store EDD by parent, then for a given scan date, just find the min EDD that is > the scan date, to generate the
    # baby id
    print("Store EDD by parent, for baby ID generation")
    edd_by_parent = defaultdict(list)

    # Where we just have one baby per parent, simple map
    num_scans = scan.groupby("patients_id").size()
    just_one_patients = num_scans[num_scans == 1].index
    just_one_edd = scan.edd[scan.patients_id.isin(just_one_patients)]

    edd_by_parent.update(dict((p, [e]) for (p, e) in zip(just_one_patients, just_one_edd)))

    # Now process those with more than one baby/scan.  Quicker to do it this way.
    more_patients = num_scans[num_scans != 1].index

    for patients_id in more_patients:
        edd_by_parent[patients_id] = sorted(scan.edd[scan.patients_id == patients_id])

    def get_baby_id(patient_id=None, scan_date=None):
        dates = edd_by_parent.get(patient_id)
        if dates is None:
            # raise Exception("Unable to find EDD data for patient {0}".format(patient_id))
            return None
        if len(dates) == 1:
            return "{0}_{1}".format(patient_id, dates[0])
        else:
            # Filter out any dates less than the scan date, then take the min of what's left (because we already
            # sorted the dates
            filter_dates = list(filter(lambda x: x >= scan_date, dates))
            if len(filter_dates) == 0:
                # Baby must be overdue, so take the most recent EDD
                baby_edd = dates[-1]
            else:
                baby_edd = filter_dates[0]

            # print(patients_id, scan_date, dates, filter_dates)
            return "{0}_{1}".format(patient_id, baby_edd)

    # TODO The estimated delivery date could be before the scan date, if the baby is overdue, so need to allow for that
    # TODO if this happens, then I think we just take the most recent estimated delivery date, it must be the right one

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
        print("First 10:")
        print("\n".join(map(str, missing_patient_ids[0:10])))

        # Need to drop those rows with missing baby ids, can't do anything with them
        scan_df = scan_df[scan_df.baby_id != None]
        return scan_df, missing_patient_ids

    if False:
        print("Process scan2")
        scan2, scan2_missing_patients = process_subsequent_scan(dfs_by_scan["scan2"], "scan2")

    if True:
        print("Process scan3")
        scan3, scan3_missing_patients = process_subsequent_scan(dfs_by_scan["scan3"], "scan3")

    # Then, for each baby, we can separate out the different data:
    # - parent/baby meta
    # - scan
    # - outcome

    #TODO Do a proper check using sets to identify missing patients, especially when we have ones in scan2 and 3 but
    # not scan 1

    # In the end we want a big stacked table, with one row per scan
    # The id is therefore the baby_id + scan_date
    # The parent data and other meta data will be present in each row, the tidying up of this data needs to happen
    # before we do the denormalisation, otherwise record counts will be inaccurate
    # At some point also need to convert dates etc, so that we can do better date logic and analysis

    # First of all, just identify the scan fields and stack those together
