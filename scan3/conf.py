from environment import *

TEST_NAME = "test name"
TEST_PROJECT = "test project"


# Scan 1+2+3 intersect == scan fields
SCAN_FIELDS = ['date_of_exam', 'bpd3', 'l_uterine_a_pi', 'l_uterine_a_ri', 'r_uterine_a_pi', 'r_uterine_a_ri']

# Scan 1-2-3 == parent info
PARENT_FIELDS = ['alcohol', 'b_hcg', 'bhcg_mom', 'cigarettes', 'conception', 'crl_1', 'edd_lmp', 'edd_us', 'episode_lmp', 'ethnic_group', 'ethnic_group2', 'first_trim_msb_date', 'height_cm', 'iud_16_23w', 'iud_24_36w', 'iud_gte_37w', 'iud_lt_15w', 'msb_manufacturer', 'msb_plgf_mom', 'msb_plgf_pgml', 'nt1', 'pappa', 'pappa_mom', 'para', 'weight_kg']

# Scan 3-2-1 == outcome info
OUTCOME_FIELDS = ['admission_scbu', 'apgar_10min', 'apgar_5min', 'bw', 'chromosomes_baby', 'comments', 'cord_ph_artery', 'cord_ph_vein', 'delivery', 'discharged_on', 'fetus_mid_cerebral_a_pi', 'fetus_umbilical_a_pi', 'fh4', 'karyotype_baby', 'outcome', 'outcome_date', 'outcome_died_on', 'outcome_ga_days', 'outcome_ga_weeks', 'postnatal_diagnosis', 'sex_of_child']

DEBUG_FIELDS = ["center", "filename"]

DROP_FIELDS = ["id_sort", "case_no", 'us_gestation_weeks1']

FIELD_SYNONYMS_BY_BASE = dict(
    pappa_mom=["pappa_a_mom"],
    us_gestation_weeks1=["us_gestation_wks1"],
    bpd3=["bpd"],
    ac3=["ac"],
    date_of_exam=["scan_date"],
    outcome_died_on=["date_of_death"],
    outcome_date=["date_of_delivery"],
    comments=["commentsgeneral_outcome"]
)

FIELD_SYNOMYMS = {}

# Make this easy to use
for bname, syns in iter(FIELD_SYNONYMS_BY_BASE.items()):
    for syn in syns:
        if syn in FIELD_SYNOMYMS:
            raise Exception("Duplicate synonym {0}->{1}".format(syn, bname))
        FIELD_SYNOMYMS[syn] = bname
