# Benefits

Benefits and participation for welfare and transfer programs are systematically underreported in the Current Population Survey (CPS) Annual Social and Economic supplement (ASEC). For some programs like Medicaid and Medicare, participation is reported but benefits are excluded entirely. The implicit marginal tax rates that stem from transfer and benefits programs are not included in the CPS.
The open-source Benefits-Augmenter model adjusts the CPS ASEC for the under-reporting of welfare and transfer program participation and benefits, imputes benefits where they are excluded, and imputes marginal tax rates that stem from welfare and transfer programs. Adjusted welfare and transfer data can serve as the basis for micro-simulating policy reforms that replace existing welfare and transform programs.

Currently this Repo includes the following programs, derived from 2014 CPS March Supplement:

- Programs with participation and benefits based on CPS ASEC:
- Supplemental Security Income (SSI): The MTR is imputed from program rules. We are also experimenting with machine learning algorithms to estimate MTR from original CPS data.
- Supplemental Nutritional Assistance Program (SNAP): The MTR is imputed from program rules.
- Veterans Benefit: No MTR is imputed since this is a social insurance program.
Social Security: The MTR is estimated for people in the work force, based on their current earnings, education experience, and work experience. The model for MTR estimation will be improved with respect to life-earning projection assumptions.

-   Programs with participation based on CPS ASEC:
- Medicaid: Participation is augmented to match administrative data. Insurance values for each participant are imputed based on Medical Expenditure Penal Survey (MEPS) expenditure data. No MTR is imputed for current version, but may be estimated at the eligible income upper bound in future.
- Medicare: Insurance values for each participant are imputed based on Medical Expenditure Penal Survey (MEPS) expenditure data. No MTR is imputed.

- Program based CPS Tax-Unit (link; footnote created by John O’hare)
- Affordable Care Act Premium Tax Credit (ACA PTC): Both participation and benefit amounts are estimated using employment, current insurance coverage, and county level residence at tax-unit level, with the ACA PTC calculator developed by Martin Holmer and Nikolai Boboshko (link to calculator).  

This Repo will include the following programs in the near future:
- Temporary Assistance for Needy Families (TANF)
- Unemployment Insurance
- General Assistance
- Public Housing
- Women, Infants and Children (WIC)
- Unemployment Insurance
- Workman’s Compensation