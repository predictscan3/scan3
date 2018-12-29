from environment import PROJECT_ROOT

from os.path import join


JOINER_VERSION = 3
NORMER_VERSION = 1

DATA_IN_ROOT = join(PROJECT_ROOT, "data")
DATA_OUT_ROOT = DATA_IN_ROOT

PARENT_FIELDS = ['alcohol', 'cigarettes', 'conception',  'edd_lmp', 'edd_us', 'edd',
                 'episode_lmp', 'ethnic_group', 'ethnic_group2', 'height_cm',
                 'para', 'weight_kg', 'dob']

# TODO: Need to update all this as we will have scan fields per file, and going to be taking the most recent one in each
# TODO: case (mostly).
# TODO: Also each scan file doesn't necessarily have the same scan fields, so will need to list them separately here.
# TODO: Means some of what I did is redundant, but the finding duplicates part isn't.
# TODO: Add some calculated fields, especially gestational age, express in days?

COMMON_FIELDS = ['patients_id', 'date_of_exam']

OUTCOME_FIELDS = ['outcome', 'outcome_date', 'outcome_died_on']

TRIM1_SCAN_FIELDS = ['date_of_exam',
                     'msb_date', 'msb_manufacturer', 'msb_plgf_pgml',
                     'bpd3',
                     'l_uterine_a_pi', 'l_uterine_a_ri', 'r_uterine_a_pi', 'r_uterine_a_ri',
                     'crl_1', 'nt1', 'b_hcg', 'pappa', ]

TRIM2_SCAN_FIELDS = ['date_of_exam',
                     'ac3', 'hc3', 'fl3', 'cord',
                     'l_uterine_a_pi', 'l_uterine_a_ri', 'r_uterine_a_pi', 'r_uterine_a_ri',
                     ]

TRIM3_SCAN_FIELDS = ['date_of_exam',
                     'ac3', 'hc3', 'fl3',
                     'l_uterine_a_pi', 'l_uterine_a_ri', 'r_uterine_a_pi', 'r_uterine_a_ri',
                     'fh4', 'fetus_mid_cerebral_a_pi', 'fetus_umbilical_a_pi',
                     ]

CHECK_FIELDS_BY_SCAN_FILE = dict(
    scan1=TRIM1_SCAN_FIELDS,
    scan2=TRIM2_SCAN_FIELDS,
    scan3=TRIM3_SCAN_FIELDS + OUTCOME_FIELDS
)

FIELDS_BY_SCAN_FILE = dict(
    scan1=TRIM1_SCAN_FIELDS,
    scan2=TRIM2_SCAN_FIELDS,
    scan3=TRIM3_SCAN_FIELDS
)

DEBUG_FIELDS = ["center", 'comments', 'scanfile_idx']

# These have useful information but we're not using them at the moment
MIGHT_NEED_FIELDS = ['iud_16_23w', 'iud_24_36w', 'iud_gte_37w', 'iud_lt_15w', 'bhcg_mom', 'pappa_mom',
                     'msb_plgf_mom', ]

# Some fields are only dropped from specific scan files
SCAN2_DROP_FIELDS = ['bpd3', 'ofd3']
SCAN3_DROP_FIELDS = ['bpd3', 'ofd3']

# Adding this separately to make it explicit where these fields are from
CENTER2_DROP_FIELDS = ['race', 'race_2', 'smoking', 'us_gestation_wks', 'chronic_hypertension', 'diabetes',
                       'diagnosis_details', 'diagnosis_scan', 'efw', 'efw_method', 'gravida', 'placenta',
                       'medical_conditions', 'medications', 'nicu_comments', 'ocd', 'other_complications_outcome',
                       'other_medical_historyixxx', 'p_v', 'pad_t', 'papp_a_mom', 'pre__weight', 'sle',
                       'smoking', 'ua', '36_weight', 'apad_2', 'aps', 'height']

DROP_FIELDS = ["id_sort", "case_no", 'admission_scbu', 'apgar_10min', 'apgar_5min',
               'outcome_ga_days', 'outcome_ga_weeks', 'postnatal_diagnosis', 'sex_of_child',
               'bw', 'chromosomes_baby', 'cord_ph_artery', 'cord_ph_vein', 'delivery', 'discharged_on',
               'karyotype_baby',
               'ad1_wb2', 'ad2_wb2',
               'maternal_age_at_exam',
               'fetus_mid_cerebral_a_pi2',
               # 'us_gestation_weeks1', 'days1', 'gestation_days',
               # Need these to "guess" an EDD to generate a baby id when there is no scan1
               ] \
              + MIGHT_NEED_FIELDS \
              + CENTER2_DROP_FIELDS

# Only rename/drop these at the final stage, they're used for various things during processing
FINAL_DEBUG_FIELDS = ['comments', 't1_mat_age_at_exam', 't2_mat_age_at_exam', 't3_mat_age_at_exam', 'patients_id']
FINAL_DROP_FIELDS = ['missing_scan_fields', 'us_gestation_weeks1', 'days1', 'filename', 'baby_id', 'center']

DROP_FIELDS_BY_SCAN = dict(
    scan1=DROP_FIELDS,
    scan2=DROP_FIELDS + SCAN2_DROP_FIELDS,
    scan3=DROP_FIELDS + SCAN3_DROP_FIELDS
)

FIELD_SYNONYMS_BY_BASE = dict(
    pappa_mom=["pappa_a_mom"],
    us_gestation_weeks1=["us_gestation_wks1", "us_gestation_wks"],
    days1=["gestation_days"],
    edd_lmp=['edd'],
    edd_us=['edd_by_us'],
    bpd3=["bpd"],
    ac3=["ac"],
    date_of_exam=["scan_date"],
    outcome_died_on=["date_of_death"],
    outcome_date=["date_of_delivery"],
    comments=["commentsgeneral_outcome"],
    msb_date=['first_trim_msb_date'],
    fl3=['fl', 'fl_3'],
    hc3=['hc', 'hc_3'],
    l_uterine_a_pi=['l__uterine_artery_pita'],
    r_uterine_a_pi=['r__uterine_artery_pita'],
    fetus_mid_cerebral_a_pi=['mca'],  # r_mca_pi # TODO Special case, need to collapse
    fetus_mid_cerebral_a_pi2=['r_mca_pi'],
    outcome_ga_days=['out_ga_d'],
    outcome_ga_weeks=['out_ga_w'],
)

FIELD_SYNOMYMS = {}

# Make this easy to use
for bname, syns in iter(FIELD_SYNONYMS_BY_BASE.items()):
    for syn in syns:
        if syn in FIELD_SYNOMYMS:
            raise Exception("Duplicate synonym {0}->{1}".format(syn, bname))
        FIELD_SYNOMYMS[syn] = bname
