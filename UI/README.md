# Unemployment Insurance Program Imputation

Unemployment Insurance (UI) compensation is reported on an individual level in the CPS.

Target data for the imputation came from official [DOL data](https://workforcesecurity.doleta.gov/unemploy/DataDownloads.asp),
<<<<<<< HEAD
called "_ETA 5195_" and "_ETA 539_". To get 2014 data for our imputation download the Workshare and Regular Program data files.
=======
called "_ETA 5195_". To get 2014 data for our imputation download the Workshare and Regular Program data files.
>>>>>>> upstream/master

We use the 2015 CPS because the UI variables, UC_YN and UC_VAL, report the respondents' unemployment
compensation and recipiency for the calendar year 2014, rather than 2015


Unemployment Insurance (UI) is intended to provide a safety net for workers who have been
displaced from work due to no fault of their own. These benefits are primarily distributed through the
State UI program, Unemployment Compensation for Federal Employees (UCFE) program, or the
Unemployment Compensation for Ex-servicemembers (UCX) program. We also include the
Workshare (STC) UI program, which allows reduced-hour employees to receive a fraction of the State
UI program benefits while working. We do not consider the Extended Benefits (EB) program, which
extends the length of coverage of the first three programs, because its program totals (both recipient
and benefit) are less than 0.1% of the regular and Workshare UI totals, and these benefits were mostly
phased out by 2014


<<<<<<< HEAD
In the calendar year 2014 (we use calendar year because CPS respondents report UI claims for calendar year), according to _ETA 5195_ and _ETA 539_ 10 million individuals claimed roughly 35.7 billion dollars in UI benefits (for all programs). The annual average combined benefit for each
recipient is approximately $3,410 with significant variation across states.
=======
In the calendar year 2014 (we use calendar year because CPS respondents report UI claims for calendar year), according to _ETA 5195_
10 million individuals claimed roughly 35.7 billion dollars in UI benefits (for all programs). The annual average combined benefit for each
recipient is approximately $3,538 with significant variation across states.
>>>>>>> upstream/master

Correspondingly, 2015 CPS totals indicate that around 4.6 million individuals claimed roughly 22.6 billion in UI benefits (all programs combined). This underreporting is typical of government benefit program questions on the CPS.


## Imputation Procedure

Recipients and benefits are imputed using a two step procedure. First, participation
is imputed to match administrative totals through an augmentation process.
Second, we impute an adjustment ratio
for benefits in order to match total benefit amounts.

Before running the imputation procedure, one must do two things (besides downloading the appropriate CPS file into your directory):

1. Open and run the _Rf\_probs.py_ script. 
- This creates the Random Forest Classifer probabilities used in the imputation labeled as _rf\_probs.csv_. 

2. Run _create\_admin.py_ with the appropriate [DOL administrative data](https://workforcesecurity.doleta.gov/unemploy/DataDownloads.asp) already downloaded into the imputation script location.  (refer to documentation for more information).
<<<<<<< HEAD
- Download _aw5159.csv_, _ar5159.csv_, and _ar539.csv_ (this last file is from ETA 539). This creates the administrative data in the file _Admin\_totals\_all.csv_ used in the imputation.
=======
- Download both _aw5159.csv_ and _ar5159.csv_. This creates the administrative data in the file _Admin\_totals\_all.csv_ used in the imputation.
>>>>>>> upstream/master

## Imputing Recipients

Since UI program rules require a base period of substantial earnings, we use the _ptotval_ variable as an income proxy. We then create a binary variable signifying all those who earn below $5200, as a proxy for those who have reached their base period earnings or not.

We follow create other variables that correspond to UI eligibility rules, as described in our documentation, to increase the predictive power of our logistic regression.

We then use the following logistic regression to determine the likelihood of an individual
participating in the program:

_UC\_YN_ = &alpha; + &beta; _weuemp_ + &beta; _ptotval_ +
                        &beta; _pruntype_ + &beta; _a\_explf_ +
                        &beta; _lkweeks_ + &beta; _lkstrch_ +
			&beta; _f\_mv\_fs_ + &beta; _disability_ + &epsilon;


Individuals are then ranked by the probability of participation. Then for each
state's subgroup we aggregate all the weights of those who are current UI recipients, and add on the most likely participants who aren't currently receiving UI, until we reach the administrative
level totals for that state.

In rf_probs.ipynb, we use a more accurate Random Forest Classifier (RFC) model, instead of a logistic regression, to determine the
likelihood of individual UI participation. We then repeat the same step above with the new probabilities.

## Benefit Imputation

For each imputed household recipient, we assign the average benefit amount for
their state. After imputation, We then calculate the total outlays for each state and
compare this total to DOL administrative data. The accuracy of our imputations
varies state-by-state, so adjustment ratios are determined for each state by
dividing total administrative benefits by total imputed benefits. Most ratios
are close to one and are used to increase each individual's
benefits.

### Continuous Updating
The following variables will need to be updated in the model and this document
when moving to the next year:
1. CPS, should be for the year after the year being considered (since UI is reported for the previous year)
2. UI income eligibility standards
3. In _create\_admin.py_, modify the year used in the code to the year considered
