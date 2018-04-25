# Medicare and Medicare Microdata

Current Population Survey (CPS) has Medicare and Medicaid coverage information available at individual level but not benefit amounts. We imputed Medicare and Medicaid benefit from Medical Expenditure Panel Survey (MEPS) for each individual currently covered by either program. In the imputation process, we aimed at targeting total benefit expenditures, excluding administrative expenses, for non-institutionalized population. 

## Total number of recipients

MEPS has already imputed coverage and benefit before distributing the datasets with their household and provider components in the survey. Thus, the number of recipients are fairly close to official reports for both programs. Medicare annual report claims it covered 53.8 million people in 2014, and MEPS Medicare recipients sum up to about 49.6 million, which is not far away since MEPS doesn’t cover institutionalized population. CPS Medicare section also has about 48.9 million recipients, so there’s not much need for imputation. But Medicaid has a slightly different story. Medicaid covers about 64 millions beneficiaries, while MEPS has about 74 million and CPS has about 54 million. MEPS seems to cover more people who enrolled for a short period of a year, so the total recipients are higher. But CPS is under-reported compared to Medicare. Again, if consider the population living inside institutions, CPS is not too far away. At this moment, we leave CPS coverage information as it is.

## Benefit

The amount of benefit is very pertinent to individual’s health condition, age, close to death, income etc. But information like how close to death is not observable, and some other information like health condition is available in MEPS but not so in CPS. Therefore, we made a compromise with the match and only execute the matching procedure based on variables available in both datasets. In current routine, we used age, gender, census region and income to match MEPS beneficiary to CPS. 

Specifically, for each CPS beneficiary, we find a pool of donors from MEPS defined by following standards:

- age within plus or minus 2 range
- same gender
- same census region
- income within plus or minus 100 range

Then we use a random number generator to pick one donor from the pool defined above, and assign this donor’s benefit amount to the corresponding CPS beneficiary. This set of standards work well for the vast majority of beneficiaries in CPS. However there’re several scenarios the method doesn’t work. First, CPS has beneficiaries older than 85, but the max age in MEPS is 85. Thus all 85 or older CPS medicare/medicaid beneficiary share the same group of donor from MEPS who are labeled at 85 years old. Second, certain age range of donors in MEPS don’t have income fell in the given range. Current routine use the donor with closest income in the given age range. 

This matching routine assumes the matched CPS benefits replicate the MEPS distribution in terms of income, age, gender, and census region. However, the aggregate benefit at this point is still well-below official spending of the year. In 2014 Medicare Trustee report, benefit expense is about 604 billion, and if deducting skilled nursing facilities in which most peopler’s institutionalized, the total non-institutionalized expense is about 576 billion. MEPS has about 426 billions total benefits, and CPS got matched up to 416 billions, which means a roughly 27% gap. Medicaid is way lower in MEPS compared to Medicare. Total benefit expense is 449 billions in 2014, and even if we approximate deduct institutional cost about 68 billions (footnote), there’s still 381 billions. However, MEPS has only 170 billion payments on file, and after matching, CPS has about 137 billion, partially due to the lower number of enrollee as well. In other words, roughly there’s a 64% gap between the micro data and official number. 

Although current goal is to target the total expense, we tried to maintain the income distribution of this micro benefit data. However, there’s not much information available online regarding this type of distribution. Instead, we applied a scaler to each individual’s benefit to augment it by same ratio to get the total official spending. This part of routine is calling for improvement in particular. In the end, the CPS matched dataset’s total number of recipients and benefits are as following:

| Program		| Medicare		| Medicaid|
| --------------------- |:---------------------:| -------:|
| Enrollees (m)		| 49			| 54      |
| Total Benefits (b)	| 576			| 381     |
