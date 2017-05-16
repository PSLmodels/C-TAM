# Benefits

Benefits and participation for welfare and transfer programs are systematically underreported in the Current Population Survey (CPS) Annual Social and Economic supplement (ASEC). For some programs like Medicaid and Medicare, participation is reported but benefits are excluded entirely. The implicit marginal tax rates that stem from transfer and benefits programs are not included in the CPS.

The CPS Transfer Augmentation Model (C-TAM) model adjusts the CPS ASEC for the under-reporting of welfare and transfer program participation and benefits, imputes benefits where they are excluded, and imputes marginal tax rates that stem from welfare and transfer programs. Adjusted welfare and transfer data can serve as the basis for micro-simulating policy reforms (for example, [Basic Income](https://github.com/open-source-economics/Benefits/blob/master/Basic%20Income.pdf)) that replace existing welfare and transfer programs.

Currently this Repo includes the following programs<sup>1</sup>, derived from 2014 CPS March Supplement:

- Programs with participation and benefits based on CPS ASEC:
    - Supplemental Security Income ([SSI](https://github.com/open-source-economics/Benefits/tree/master/SSI)): The MTR is imputed from program rules. We are also experimenting with machine learning algorithms to estimate MTR from original CPS data.
    - Supplemental Nutritional Assistance Program ([SNAP](https://github.com/open-source-economics/Benefits/tree/master/SNAP)): The MTR is imputed from program rules.
    - [Veterans Benefit](https://github.com/open-source-economics/Benefits/tree/master/VB): No MTR is imputed since this is a social insurance program.
    - [Social Security](https://github.com/open-source-economics/Benefits/tree/master/SS): The MTR is estimated for people in the work force, based on their current earnings, education experience, and work experience. The model for MTR estimation will be improved with respect to life-earning projection assumptions.

-   Programs with participation based on CPS ASEC:
    - Medicaid: Participation is augmented to match administrative data. Insurance values for each participant are imputed based on Medical Expenditure Penal Survey (MEPS) expenditure data. No MTR is imputed for current version, but may be estimated at the eligible income upper bound in future.
    - Medicare: Insurance values for each participant are imputed based on Medical Expenditure Penal Survey (MEPS) expenditure data. No MTR is imputed.

- Program based CPS Tax-Unit<sup>2</sup>
    - Affordable Care Act Premium Tax Credit (ACA PTC): Both participation and benefit amounts are estimated using employment, current insurance coverage, and county level residence at tax-unit level, with the ACA PTC calculator developed by Martin Holmer and Nikolai Boboshko<sup>3</sup>.  

This Repo will include the following programs in the near future:

- Temporary Assistance for Needy Families (TANF)
- Unemployment Insurance
- General Assistance
- Public Housing
- Women, Infants and Children (WIC)
- Unemployment Insurance
- Workman’s Compensation


1. Both code scripts and documentation in PDF format are included in this Repository. For editable documentation, please contact Amy Xu at amy.xu@aei.org.
2. This dataset is created by John O'hare from Quantria Strategies. Documentation can be found on Quantria [website](http://www.quantria.com/assets/img/TechnicalDocumentationV4-2.pdf).
3. OSPC's ACA PTC [calculator](http://chiselapp.com/user/mrh/repository/OSPC-ACA/doc/trunk/www/home.wiki).