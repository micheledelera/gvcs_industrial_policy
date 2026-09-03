*************************************************************************************************************************************************************
/*Do-file for the paper "On the scope of PTAs and GVC trade"
Date: 02.02.2021*/
*************************************************************************************************************************************************************
clear all
cd "C:\Users\miche\OneDrive\Documenti\Michele\UNU-MERIT\1_Thesis\Paper 2_PTAs and GVCs\2_Data\1_Data"
use "paper2_ptas and gvcs_database_n", clear
*************************************************************************************************************************************************************
*Recode variables for legal enforcement and generate variables for the analysis*
*************************************************************************************************************************************************************
/*
egen core_breadth_le = rowtotal(wto_plus_ftaindustrial_le wto_plus_ftaagriculture_le wto_plus_customs_le wto_plus_exporttaxes_le wto_plus_sps_le wto_plus_tbt_le wto_plus_ad_le wto_plus_cvm_le wto_plus_trims_le wto_plus_trips_le wto_x_movementofcapital_le wto_plus_ste_le wto_plus_stateaid_le wto_plus_gats_le wto_x_labourmarketregulation_le wto_plus_publicprocurement_le wto_x_ipr_le wto_x_competitionpolicy_le)
egen border_le = rowtotal(wto_plus_ftaindustrial_le wto_plus_ftaagriculture_le wto_plus_customs_le wto_plus_exporttaxes_le wto_plus_sps_le wto_plus_tbt_le wto_plus_ad_le wto_plus_cvm_le wto_plus_trims_le wto_plus_trips_le wto_x_movementofcapital_le)
egen behind_border_le = rowtotal(wto_plus_ste_le wto_plus_stateaid_le wto_plus_gats_le wto_x_labourmarketregulation_le wto_plus_publicprocurement_le wto_x_ipr_le wto_x_competitionpolicy_le wto_x_investment_le)
*/
*************************************************************************************************************************************************************
*Prepare the database: generate logged variables, xtset, and generate time, exporter_time, and importer_time dummies*
*************************************************************************************************************************************************************
local vars "capital intermediate generic custom"
foreach x of local vars {
gen ln_`x' = ln(`x')
}

label var intermediate "Intermediate"
label var generic "Generic"
label var custom "Customized" 
label var ln_intermediate "Intermediate exports"
label var ln_custom "Customized exports"
label var ln_generic "Non-customized exports" 

xtset pair_id year

rename wto_x_labourmarketregulation_le wto_x_labourmarket_le
foreach var of varlist pta bit dtt wto_plus_ftaindustrial_le wto_plus_ftaagriculture_le wto_plus_customs_le wto_plus_exporttaxes_le wto_plus_sps_le wto_plus_tbt_le wto_plus_ad_le wto_plus_cvm_le wto_plus_trims_le wto_plus_trips_le wto_x_movementofcapital_le wto_plus_ste_le wto_plus_stateaid_le wto_plus_gats_le wto_x_labourmarket_le wto_plus_publicprocurement_le wto_x_ipr_le wto_x_competitionpolicy_le wto_x_investment_le incomegroup_code_o incomegroup_code_d{
  replace `var' = `var'[_n-1] if missing(`var') 
}
foreach var of varlist wto_plus_ftaindustrial_le wto_plus_ftaagriculture_le wto_plus_customs_le wto_plus_exporttaxes_le wto_plus_sps_le wto_plus_tbt_le wto_plus_ad_le wto_plus_cvm_le wto_plus_trims_le wto_plus_trips_le wto_x_movementofcapital_le wto_plus_ste_le wto_plus_stateaid_le wto_plus_gats_le wto_x_labourmarket_le wto_plus_publicprocurement_le wto_x_ipr_le wto_x_competitionpolicy_le wto_x_investment_le{
  gen `var'_n = 0 
  replace `var'_n = 1 if `var' == 2
}

drop north2north north2south south2south south2north
gen north2north = 0
replace north2north = 1 if incomegroup_code_o == "HIC" & incomegroup_code_d == "HIC"
gen north2south = 0
replace north2south = 1 if incomegroup_code_o == "HIC" & (incomegroup_code_d == "LIC" | incomegroup_code_d == "MIC")
gen south2north = 0
replace south2north = 1 if (incomegroup_code_o == "LIC" | incomegroup_code_o == "MIC") & incomegroup_code_d == "HIC"
gen south2south = 0
replace south2south = 1 if (incomegroup_code_o == "LIC" | incomegroup_code_o == "MIC") & (incomegroup_code_o == "LIC" | incomegroup_code_o == "MIC")
gen income = 0
replace income = 1 if north2south==1
replace income = 2 if south2south==1
replace income = 3 if south2north==1
replace income = 0 if north2north==1

drop if year < 1995

gen pta_inv = 0
replace pta_inv = 1 if pta==1 & wto_x_investment_le_n==1

tab(year), gen(dumy)
egen exp_time = group(iso_o year)
egen imp_time = group(iso_d year)
encode iso_o, gen(origin)
encode iso_d, gen(destination)

cd "C:\Users\miche\OneDrive\Documenti\Michele\UNU-MERIT\1_Thesis\Paper 2_PTAs and GVCs\2_Data\2_Results"
*************************************************************************************************************************************************************
*Summary statistics on all sample*
*************************************************************************************************************************************************************
sum pta wto_x_investment_le_n bit dtt expval generic custom north2north north2south south2north south2south  
*************************************************************************************************************************************************************
*Regression analysis (1): The effect of legally enforceable investment provisions in PTAs at different levels of income, controlling for BITs and other provisions*
*************************************************************************************************************************************************************
eststo: ppmlhdfe expval pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
eststo: ppmlhdfe expval pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
eststo: ppmlhdfe generic pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
eststo: ppmlhdfe generic pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
eststo: ppmlhdfe custom pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
eststo: ppmlhdfe custom pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
esttab using gravity_baseline2.tex, b se star (* 0.10 ** 0.05 *** 0.01) mtitles("All" "All" "Generic" "Generic" "Customized" "Customized") booktabs
esttab using gravity_2018.csv, star (* 0.10 ** 0.05 *** 0.01) replace 
eststo clear 
*************************************************************************************************************************************************************
*Regression analysis (2): Mechanisms: what is the role of institutions, and particularly the rule of law?*
*************************************************************************************************************************************************************
eststo: ppmlhdfe expval pta wto_x_investment_le_n##income bit dtt rle wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n, absorb(origin#destination origin#year year) vce(cluster pair_id)
eststo: ppmlhdfe generic pta wto_x_investment_le_n##income bit dtt rle wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n, absorb(origin#destination origin#year year) vce(cluster pair_id)
eststo: ppmlhdfe custom pta wto_x_investment_le_n##income bit dtt rle wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n, absorb(origin#destination origin#year year) vce(cluster pair_id)
esttab using gravity_inst.tex, b se star(* 0.10 ** 0.05 *** 0.01) mtitles("All" "Generic" "Customized") booktabs
esttab using gravity2_2018.csv, star (* 0.10 ** 0.05 *** 0.01) replace
eststo clear
*************************************************************************************************************************************************************
*Regression analysis (3): Channels: what happens to high tech and complex goods?*
*************************************************************************************************************************************************************
eststo: ppmlhdfe high_tech pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
eststo: ppmlhdfe complex_95 pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
esttab using gravity_channels.tex, b se star (* 0.10 ** 0.05 *** 0.01) mtitles("High-tech" "Complex") booktabs
esttab using gravity_2018.csv, star (* 0.10 ** 0.05 *** 0.01) replace 
*************************************************************************************************************************************************************
*Regression analysis (4): Leads and lags
*************************************************************************************************************************************************************
ppmlhdfe custom pta F2.wto_x_investment_le_n##income F1.wto_x_investment_le_n##income wto_x_investment_le_n##income L1.wto_x_investment_le_n##income L2.wto_x_investment_le_n##income L3.wto_x_investment_le_n##income L4.wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)

*by subsample (note to self, only south-north is significant, not north-south) as robustness, as requested by Neil
ppmlhdfe custom pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n if north2north==1, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)

ppmlhdfe custom pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n if south2north==1, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)

ppmlhdfe custom pta wto_x_investment_le_n##income bit dtt wto_plus_trims_le_n wto_plus_trips_le_n wto_x_ipr_le_n wto_plus_ftaindustrial_le_n wto_plus_ftaagriculture_le_n wto_plus_customs_le_n wto_plus_exporttaxes_le_n wto_plus_sps_le_n wto_plus_tbt_le_n wto_plus_ad_le_n wto_plus_cvm_le_n  wto_x_movementofcapital_le_n wto_plus_ste_le_n wto_plus_stateaid_le_n wto_plus_gats_le_n wto_x_labourmarket_le_n wto_plus_publicprocurement_le_n wto_x_competitionpolicy_le_n if south2south==1, absorb(origin#destination origin#year destination#year year) vce(cluster pair_id)
