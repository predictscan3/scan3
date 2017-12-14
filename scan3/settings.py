from scan3.conf import *

# Developer specific settings and configuration here
# Most things should be common though, so should be in conf.py

# Here is an example of overwriting a setting
TEST_NAME = "this is overwritten"

# Scan 1+2+3 intersect == scan fields
SCAN_FIELDS = ['date_of_exam', 'bpd3', 'l_uterine_a_pi', 'l_uterine_a_ri', 'r_uterine_a_pi', 'r_uterine_a_ri', 'us_gestation_weeks1']
# Scan 1-2-3 == parent info
PARENT_FIELDS = ['alcohol', 'b_hcg', 'bhcg_mom', 'cigarettes', 'conception', 'crl_1', 'edd_lmp', 'edd_us', 'episode_lmp', 'ethnic_group', 'ethnic_group2', 'first_trim_msb_date', 'height_cm', 'iud_16_23w', 'iud_24_36w', 'iud_gte_37w', 'iud_lt_15w', 'msb_manufacturer', 'msb_plgf_mom', 'msb_plgf_pgml', 'nt1', 'pappa', 'pappa_mom', 'para', 'weight_kg']
# Scan 3-2-1 == outcome info
OUTCOME_FIELDS = ['admission_scbu', 'apgar_10min', 'apgar_5min', 'bw', 'chromosomes_baby', 'comments', 'cord_ph_artery', 'cord_ph_vein', 'delivery', 'discharged_on', 'fetus_mid_cerebral_a_pi', 'fetus_umbilical_a_pi', 'fh4', 'karyotype_baby', 'outcome', 'outcome_date', 'outcome_died_on', 'outcome_ga_days', 'outcome_ga_weeks', 'postnatal_diagnosis', 'sex_of_child']
