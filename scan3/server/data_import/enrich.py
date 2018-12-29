from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def to_datetime(s=None):
    """
    Missing or dodgy dates are returned as `NaT`s
    :param s:
    :return:
    """
    return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")


def convert_dates(df=None):
    date_fields = ["t1_date_of_exam", "t2_date_of_exam", "t3_date_of_exam",
                   "t1_msb_date",
                   "dem_dob", "dem_edd", "dem_edd_lmp", "dem_edd_us",
                   "outcome_date", "outcome_died_on"]
    for f in date_fields:
        df[f] = to_datetime(df[f])
    return df


def add_scan_calcd_fields(prefix=None, scan_df=None):
    """
    Convert fields in-place to dates and work out the mothers age at scan date, to help with matching subsequent scans
    :param prefix:
    :param scan_df:
    :return:
    """
    prefix = prefix + "_" if prefix is not None else ""

    scan_df["dob"] = to_datetime(scan_df.dob)
    scan_df["date_of_exam"] = to_datetime(scan_df.date_of_exam)
    if "outcome_date" in scan_df.keys():
        scan_df["outcome_date"] = to_datetime(scan_df.outcome_date)
    scan_df[prefix + "mat_age_at_exam"] = (scan_df.date_of_exam - scan_df.dob) / np.timedelta64(1, "Y")

    return scan_df


def add_calcd_fields(df=None):
    # Need gestational age in days (and weeks?)
    # Check notes, but might also be useful to express gestational age as a
    # percentage of 40 weeks, can be > 1 if overdue
    # Also the mothers age
    # Think need to work out the conception date from via EDD - 40 weeks?
    df["debug_ecd"] = df.dem_edd - timedelta(weeks=40)

    # Note that this doesn't account for there being no baby in the later scans, so the values might be misleading
    df["t1_ga_weeks"] = (df.t1_date_of_exam - df.debug_ecd) / np.timedelta64(1, "W")
    df["t2_ga_weeks"] = (df.t2_date_of_exam - df.debug_ecd) / np.timedelta64(1, "W")
    df["t3_ga_weeks"] = (df.t3_date_of_exam - df.debug_ecd) / np.timedelta64(1, "W")

    # df["t1_age"] = (df.t1_date_of_exam - df.dem_dob) / np.timedelta64(1, "Y")
    # df["t2_age"] = (df.t2_date_of_exam - df.dem_dob) / np.timedelta64(1, "Y")
    # df["t3_age"] = (df.t3_date_of_exam - df.dem_dob) / np.timedelta64(1, "Y")

    return df


def apply_filters(df=None):
    # drop rows where miscarriage < 24 weeks, or mark them as ok
    return df

