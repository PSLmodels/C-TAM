# Housing Assistance Program Imputation

Federal housing rental assistance is reported on a household, rather than
individual level in the CPS. However, the actual benefit, and its corresponding outlay, are received at the family level.

Target data for the imputation came from official [HUD data](https://www.huduser.gov/portal/datasets/assthsg.html),
called "_Picture of Subsized Households_". To get 2014 data go to the bottom of the landing page and select year 2014,
at the state summary level, with the summary of all programs and all variables. 

The Housing Choice Voucher (HCV), Section 8 Project-Based Rental Assistance (PBRA), and public housing programs are the largest federal rental assistance programs administered by The Department of Housing and Urban Development (HUD). Smaller programs that we also consider include Moderate  Rehabilitation (project-based), Section 221 Below Market Interest Rate (BMIR) (project-based), Section 202 Supportive Housing for the Elderly Program (project-based), and Section 811 Supportive Housing for Persons with Disabilities (project-based). For simplification, we lump these smaller programs into the PBRA program. These programs are intended to subsidize monthly rent payments for low-income families through long or short-term subsidy contracts for either (depending on the program) the tenant or owner of privately owned housing units, or through publicly owned housing units with lowered rents. The CPS provides micro-data for public housing with its HPUBLIC variabe, and micro-data for HCV and PBRA with its HLORENT variable. However, due to CPS respondents' inability to differentiate between these two housing assitance questions (see _Housing\_Imputation\_Report.pdf_ for details), we lumped them into a new variable "HOUSING" (if they receive either variable, or any type of rental assistance, HOUSING = 1).

In Fiscal Year 2014, according to _Picture of Subsidized Households_ around 4.5 million families claimed
roughly 36 billion dollars in federal housing assistance benefits (for all programs combined). More specifically, approximately 2.1 million families claimed roughly 16 billion dollars in HCV benefits, approximately 1.2 million families claimed roughly 10 million dollars in PBRA benefits (without smaller programs), while approximately 1 million families lived in public housing which subsidized approximately 5.8 million dollars of rent. The remaining participants and outlays come from the smaller programs listed above.

## Imputation Procedure

Recipients and benefits are imputed using a two step procedure. First, participation
is imputed to match administrative totals through an augmentation or reduction process (depending on over or underreporting). For most states we reduce this benefit's recipient totals because housing assistance is overreported in the CPS due to respondent confusion with the housing assistant questions (see pdf documentation for details).
Second, we impute an adjustment ratio
for benefits in order to match total benefit amounts.

Before running the imputation procedure, one must do five things (besides downloading the appropriate CPS file into your directory):

1. Choose whether or not to use the Supplemental Poverty Measure (SPM) dataset's 
housing subsidy amounts by setting the imputation script variable "use_spm_data = True/False" according to this decision. 
- Refer to the documentation _Housing\_Imputation\_Report.pdf_ for more details on this dataset. 

2. If you choose True, then download the [SPM 2013 dataset](https://www.census.gov/data/datasets/2013/demo/supplemental-poverty-measure/spm.html) into the same directory as the imputation script. 

3. Open the _rf\_probs.ipynb_ script and set use\_spm\_data = True/False depending on what you chose above. Run the script. 
- This creates the Random Forest Classifer probabilities used in the imputation labeled as _rf\_probs(\_spm).csv_. 

4. Run _create\_admin.py_ with the appropriate [HUD administrative data](https://www.huduser.gov/portal/datasets/assthsg.html) already downloaded into the imputation script location (refer to documentation for more information).
- This creates the administrative data in the file _Admin\_totals\_all.csv_ used in the imputation.

5. Run _create\_incomelims.py_ with the appropriate [HUD income limits](https://www.hudexchange.info/programs/home/home-income-limits/?filter_Year=2014&filter_=Scope=&filter_State=&programHOME&group=IncomeLmts) already downloaded into imputation script destination (refer to documentation for more information).
- This creates the income level data in the file _Income\_limits.csv_ used in the logistic regression (variables under30inc & under50inc).

## Imputing Recipients

We initially calculate net income according to program rules:

1. Gross income for each household is attained by summing up the earned and
unearned income of all household members.
2. An exclusion of all dependent children's income is applied (younger than 18).
3. An exclusion of all educational financial assistance is applied (we don't initially include this).
4. A cap of $480 for all children 18 and older that are full-time students is applied.

We use this definition of income as a proxy for defining the income limit threshold indicators.

We then use the following logistic regression to determine the likelihood of a household
participating in the program:

_housing indicator_ = &alpha; + &beta; _family size_ + &beta; _under30inc_ +
                        &beta; _under50inc_ + &beta; _disability_ +
                        &beta; _elderly_ + &beta; _citizenship_ +
			&beta; _ffoodst_ + &beta; _hfoodst_ +
                        &beta; _FMOOP_ + &beta; _medicaid_ + &beta; _propval_ + &epsilon;


Households are then ranked by the probability of participation. Then if we are augmenting recipients, for each
state's subgroup we aggregate all weights of the most likely participants, until they reach the administrative
level totals for that state. If we are reducing, then for each
state's subgroup we aggregate all current recipient weights until they reach the administrative
level totals for that state, then exclude the rest of the recipients from the recipient pool.

In rf_probs.ipynb we use a more accurate Random Forest Classifier (RFC) model, instead of a logistic regression, to determine the
likelihood of a houshold participating in the housing program. We then repeat the same step above with the new probabilities.

## Benefit Imputation

For each imputed household recipient, we assign the average benefit amount for
their state. After imputation, We then calculate the total outlays for each state and
compare this total to HUD administrative data. The accuracy of our imputations
varies state-by-state, so adjustment ratios are determined for each state by
dividing total administrative benefits by total imputed benefits. Most ratios
are close to one and are used to either increase or shrink each household's
benefits.

### Continuous Updating
The following variables will need to be updated in the model and this document
when moving to the next year:
1. All files downloaded (income limits, CPS, SPM, administrative data) should be for the year considered
2. Housing income eligibility standards
