from collections import defaultdict


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
    # Base this on the LMP because hoping that won't change if this baby has multiple scans in the first file
    scan["baby_id"] = scan["patients_id"].astype(str) + "_" + scan["ldd_lmp"]

    # Need a lookup of scan date to baby id, so we can work out way back to the baby id from a parent and scan date
    # Might need to do this within the parent grouping, need multiple levels anyway
    # baby_by_scan = dict()
    # for scan_date, data in scan.groupby("baby_id"):
    #     if len(data) > 1:
    #         raise Exception("Found {0} babies for scan {1}".format(len(data), scan_date))
    #     baby_by_scan[data.date_of_exam[0]] = baby

    # Get a lookup of scan dates by mother, so we can cope with the subsequent scans where a mother has multiple babies
    scan_by_parent = defaultdict(list)
    for parent, data in scan.groupby("patients_id"):
        scan_by_parent[parent] = sorted(set(data.date_of_exam))

    def get_baby_id(patient_id=None, scan_date=None):
        dates = scan_by_parent.get(patient_id)
        if dates is None:
            raise Exception("Unable to find scan1 data for patient {0}".format(patient_id))
        if len(dates) == 1:
            return "{0}_{1}".format(patient_id, dates[0])
        else:
            # Filter out any dates greater than the scan date, then take the max of what's left (because we already
            # sorted the dates
            dates = filter(lambda x: x < scan_date, dates)
            return "{0}_{1}".format(patient_id, dates[-1])

    # Now can process the next scan
    scan2 = dfs_by_scan["scan2"]
    scan2_baby_ids = []
    for row in scan2.iterrows():
        scan2_baby_ids.append(get_baby_id(row[1].patients_id, row[1].date_of_exam))
    scan2["baby_id"] = scan2_baby_ids

    pass

    # for scan, df in dfs_by_scan.iteritems():
    #     pass

    # Then, for each baby, we can separate out the different data:
    # - parent/baby meta
    # - scan
    # - outcome

