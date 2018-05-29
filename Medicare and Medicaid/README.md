# Summary

The imputation of Medicare and Medicaid is different from other programs in that the total number of beneficiaries is taken from the Current Population Survey (CPS) as-is, and benefits are matched from similar MEPS beneficiaries by age, gender, census region and income. Each benefit amount is scaled up by a constant per program to match administrative totals.

This document serves as a supplement to the main documentation, [C-TAM documentation v0.1.3](https://github.com/open-source-economics/C-TAM/blob/master/C-TAM%20Documentation%20v0.1.3.pdf), and provides necessary infomation to update the notebooks for the Medicare and Medicaid imputations. Please refer to the main documentation for more details.


## Medicare and Medicaid Microdata

The CPS has Medicare and Medicaid coverage information available at individual level but not benefit amounts. We impute Medicare and Medicaid benefit amounts from the Medical Expenditure Panel Survey (MEPS) for each individual covered by either program in that survey year. We use the full year consolidated data file (HC171; 2014), obtained through the [MEPS website](https://meps.ahrq.gov/data_stats/download_data_files_results.jsp?cboDataYear=All&cboDataTypeY=1%2CHousehold+Full+Year+File&buttonYearandDataType=Search&cboPufNumber=All&SearchTitle=Consolidated+Data).


In the imputation process, we aim at targeting total benefit expenditures, excluding administrative expenses, for the non-institutionalized population. We obtain enrollee and benefit targets from the [Medicaid Actuary Report](https://www.medicaid.gov/medicaid/finance/actuarial-report/index.html) and the [Medicare Trustee's Report](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/ReportsTrustFunds/index.html)

## Total number of recipients

The MEPS has already imputed coverage and benefit before distributing the datasets with their household and provider components in the survey. Thus, the number of recipients is fairly close to official reports for both programs. Medicare Trustee's Report claims it covered 53.8 million people in 2014, and the MEPS Medicare recipients sum up to about 49.6 million, which is not far away since the MEPS doesn't cover institutionalized population. The medicare section of the CPS also has about 48.9 million recipients, so there's not much need for imputation. But Medicaid has a slightly different story. The Medicaid Actuary Report estimates it covers about 64 millions people (point in time during FY 2014), while the MEPS has about 74 million and CPS has about 54 million. The MEPS seems to cover more people who enrolled for a short period of a year, so the total recipients are higher. But CPS is under-reported compared to Medicare. Again, if consider the population living inside institutions, CPS is not too far away. At this moment, we leave CPS coverage information as it is.

## Benefit

The amount of benefit is very pertinent to individual's health condition, age, mortality risk, income etc. But some information like health condition is available in the MEPS but not so in CPS. Therefore, we made a compromise with the match and only execute the matching procedure based on variables available in both datasets: age, gender, census region and income. 

Specifically, for each CPS beneficiary, we find a pool of donors from MEPS defined by following standards:

- age within two years
- same gender
- same census region
- income within $100

Then we use a random number generator to pick one donor from the pool defined above, and assign this donor's benefit amount to the corresponding CPS beneficiary. This set of standards works well for the vast majority of beneficiaries in the CPS. However there're several scenarios the method fails. First, the CPS has beneficiaries older than 85, but the max age in the MEPS is 85. Thus all CPS medicare/medicaid beneficiaries older than 85 share the same group of donors from the MEPS who are labeled at 85 years old. Second, some donor age ranges in the MEPS don't have incomes in the given range. So we relaxed the income constraint and use the donor with the closest income in the given age range. 

This matching routine assumes the matched CPS benefits replicate the MEPS distribution in terms of income, age, gender, and census region. However, the aggregate benefit at this point is still well-below official spending of the year. In 2014 Medicare Trustee report, benefit expense is about 604 billion, and if deducting skilled nursing facilities in which most people are institutionalized, the total non-institutional expense is about 576 billion (the Medicare Trustee's Report; Table II B1). The MEPS has about 426 billions total benefits, and the CPS beneficiary records got matched up to 416 billions, which means a roughly 27% gap. Medicaid is way lower in the MEPS compared to Medicare. The total benefit expense is 449 billions in 2014 (Table 1), and even if we approximate deduct institutional cost about 71 billions<sup>1</sup>, there's still 381 billions. However, the MEPS has only 170 billion payments on file, and after matching, the CPS has about 137 billion, partially due to the lower number of beneficiary as well. In other words, roughly there's a 64% gap between the micro data and official number. 

While the current goal is to target the total expense, we try to maintain the income distribution of this micro benefit data. However, there's not much information available online regarding this type of distribution. Instead, we augment each individual's benefit by a constant ratio to match the total official spending as outlined in the paragraph above. This leaves room for improvement. After matching, the total number of recipients (original) and benefits (matched) in the CPS are as following:

| Program		| Medicare		| Medicaid|
| --------------------- |:---------------------:| -------:|
| Beneficiary (m)	| 49			| 54      |
| Total Benefits (b)	| 576			| 397     |

Footnote:

[1] This is imputed as 15% of long-term service and supports expenditure. Please refer to footnote 72 in the general documentation for more information. 