# Benefits

Benefits and participation for welfare and transfer programs are systematically underreported in the Current Population Survey (CPS) Annual Social and Economic supplement (ASEC). For some programs like Medicaid and Medicare, participation is reported but benefits are excluded entirely. The implicit marginal tax rates that stem from transfer and benefits programs are not included in the CPS.

The CPS Transfer Augmentation Model (C-TAM) model adjusts the CPS ASEC for the under-reporting of welfare and transfer program participation and benefits, imputes benefits where they are excluded, and imputes marginal tax rates that stem from welfare and transfer programs. Adjusted welfare and transfer data can serve as the basis for micro-simulating policy reforms (for example, [Basic Income](https://github.com/open-source-economics/Benefits/blob/master/Basic%20Income.pdf)) that replace existing welfare and transfer programs.

Note: when processing the raw CPS files from NBER, use the provided STATA scripts, rather than the SAS scripts.

Currently this Repo includes the following programs<sup>1</sup>, derived from 2014 CPS March Supplement:

- Programs with participation and benefits based on CPS ASEC:
    - Supplemental Security Income ([SSI](https://github.com/open-source-economics/Benefits/tree/master/SSI)): The MTR is imputed from program rules. We are also experimenting with machine learning algorithms to estimate MTR from original CPS data.
    - Supplemental Nutritional Assistance Program ([SNAP](https://github.com/open-source-economics/Benefits/tree/master/SNAP)): The MTR is imputed from program rules.
    - [Veterans Benefit](https://github.com/open-source-economics/Benefits/tree/master/VB): No MTR is imputed since this is a social insurance program.
    - [Social Security](https://github.com/open-source-economics/Benefits/tree/master/SS): The MTR is estimated for people in the work force, based on their current earnings, education experience, and work experience. The model for MTR estimation will be improved with respect to life-earning projection assumptions.
    - [Housing Assistance](https://github.com/open-source-economics/C-TAM/tree/master/Housing): We augment data based on three major federal programs, and provides two inputation results. One imputation is based on the market value of housing subsidy estimated by Census, and the other imputation is based on the estimation from Supplemental Poverty Measure.
    - [Unemployment Insurance](https://github.com/open-source-economics/C-TAM/tree/master/UI): Imputation is based on total number of beneficiaries covered in 2014. 
    - Temporary Assistance for Needy Families (TANF)
    - Workman’s Compensation: This is not a program funded through goverments. Thus we did the imputation but have not included it in the general documentation.


-   Programs with participation based on CPS ASEC:
    - Medicaid: Participation is augmented to match administrative data. Insurance values for each participant are imputed based on Medical Expenditure Penal Survey (MEPS) expenditure data. No MTR is imputed for current version, but may be estimated at the eligible income upper bound in future.
    - Medicare: Insurance values for each participant are imputed based on Medical Expenditure Penal Survey (MEPS) expenditure data. No MTR is imputed.
    - Women, Infants and Children (WIC): Participation is edited first and augmented based on each category (Women, infants or children) of administrative data. Benefit is assigned based on state-level administrative average per person.
    

- Program based CPS Tax-Unit<sup>2</sup>
    - Affordable Care Act Premium Tax Credit (ACA PTC): Both participation and benefit amounts are estimated using employment, current insurance coverage, and county level residence at tax-unit level, with the ACA PTC calculator developed by Martin Holmer and Nikolai Boboshko<sup>3</sup>.  


Current Work-In-Progress:

- Benefit extrpolation: see more details in taxdata issue #, and Tax-Calculator PR #1500
- Minor adjustments to TANF and
- Impute institutional population from American Community Survey population

Intermediate term improvements (2017):

- Add administrative costs for all welfare programs
- Impute immigrant status
- Estimate Participation Tax Rate (PTR) for welfare participates

Long term improvements (2018):

- Model Medicaid cliff marginal tax rate
- Expand the database to include 2015 CPS


[C-TAM Documentation](https://docs.google.com/document/d/1CIfp8KwECJa4bIF9U3hHTf3P7Y19ya2NI5QhEDOyG98/edit?usp=sharing) (Latest Version, Work in Progress)

Core Maintainer

- Matt Jensen
- Anderson Frailey
- Amy Xu


Contributor

- Parker Rogers
- Yuying Wang
- James Olmstead
- Xueliang Wang

We would like to thank Dan Feenberg, Kevin Corinth, Jamie Hall and many other  scholars at the American Enterprise Institute for their thoughtful suggestions. 


Note:
1. Both code scripts and documentation in PDF format are included in this Repository. For editable documentation, please contact Amy Xu at amy.xu@aei.org.
2. This dataset is created by John O'hare from Quantria Strategies. Documentation can be found on Quantria [website](http://www.quantria.com/assets/img/TechnicalDocumentationV4-2.pdf).
3. OSPC's ACA PTC [calculator](http://chiselapp.com/user/mrh/repository/OSPC-ACA/doc/trunk/www/home.wiki).
