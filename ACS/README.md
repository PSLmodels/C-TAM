# Group Quarters Institutionalized Population Imputation

The Current Population Survey (CPS) does not include group quarters (GQ) insitutionalized individuals. GQ insitutions include correctional facilities, nursing homes, and mental hospitals. Individuals living within these insitutions receive a significant amount of government benefits, and may be important when considering program reforms. 

On the other hand, the American Community Survey (ACS) does include GQ institutionalized popultation, and many of their important demographic characteristics. Thus, we transfer these institutionalized individuals to the CPS dataset with the following corresponding CPS variables.

-'a_maritl' - Marital Status

-'ssi_val' - Value of Supplemental Security Income payments

-'ss_val' - Value of Social Security payments

-'paw_val' - Value of Public Assistance payments

-'hfoodsp' - Food Stamp Participation

-'prdtrace' - Race

-'gestfips' - State of Residency 

-'semp_val' - Self employment value (this also includes farm income)

-'wsal_val' - Wage/salary income

-'marsupwt' - Supplemental weights

-'ptotval' - Total personal income

-'pothval' - Total other income (unearned)

-'pearnval' - Total earned income (wsal\_val + frse\_val + semp\_val)

-'a_sex' - Gender

-'a_age' - Age

-'a_hga' - Educational attainment

We also include which type of institution these individuals belong to.

-'prisoner' - Those who are residing in correctional facilities

-'other_institutionalized' - Those residing outside of correctional facilities

### Data Needed Before Running Script

One must download the [ACS 1-year PUMS datasets](https://www.census.gov/programs-surveys/acs/data/pums.html) for the year being considered. Both the US Population Records and US Housing Unit Records should be downloaded for that year. Both the population and housing records will have two parts (a and b) due to the size of the data. Thus, you should have 4 seperate data files once the downloads have finished. Run the script after putting these data into the same folder as the script.
