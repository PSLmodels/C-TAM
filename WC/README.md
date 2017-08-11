# Worker's Compensation Program Imputation

Worker's Compensation (WC) is reported on an individual level in the CPS.

Target data for the imputation came from an official [SSA/NASI report](https://www.nasi.org/research/2016/workers-compensation-benefits-coverage-costs),
called "_Worker's Compensation: Benefits, Coverage, and Costs_". To get 2014 data for our imputation get the total benefits claimed amount from Table 1 in the report, and the WC
recipient total from Table 18. NASI's table 18 contains data gathered from NCCI's Annual Statistical
Bulletin. Table 18 reports the total recipients/claims amount reported per 100,000 insured workers;
SSA gives the total number of insured workers in their Annual Statistical Supplement, which was 129
million in 2014. Using this, we converted the per 100,000 recipients total from the NASI report, to the
national recipient total. These totals are hard-coded into the WC_impute.py script, but the 
claims per 100,000 insured comes from claims_projected, which is calculated in arma.ipynb and included in this repository. 
Even though these WC totals are not state level specific, we use them since they seem
to be the best estimate for WC administrative totals available; thus, we do not impute on the state
level, rather on the national level using the NASI data.
We had to project claims to 2014 since we did not have 2014 claim data from NASI. For more information on how we projected these claims, and data origins please see documentation _WC\_Impuation\_Report.pdf_.

We use the 2015 CPS because the WC variables, WC\_YN and WC\_VAL, report the respondents' worker's
compensation and recipiency for the calendar year 2014, rather than 2015.


Worker's Compensation (WC) is intended to provide a safety net for workers who are injured or
become sick on the job, by compensating for lost wages due to work-related injury or illness events,
and medical coverage for these same events. These benefits are distributed through private carriers,
state funds, and self-insured firms, with each state having its own reporting requirements and
practices.


We use calendar year administrative totals because the CPS WC compensation is reported according
to calendar year. In the calendar year 2014, administrative data suggests that approximately 3.9
million individuals claimed roughly $62.3 billion dollars in WC benefits. The annual average benefit
for each recipient is approximately $15,833.

Correspondingly, 2015 CPS totals indicate that around 955,469 individuals claimed roughly $9.8 billion in WC benefits. This underreporting is typical of government benefit program questions on the CPS, although this is rather extreme.


## Imputation Procedure

Recipients and benefits are imputed using a two step procedure. First, participation
is imputed to match administrative totals through an augmentation process.
Second, we impute an adjustment ratio
for benefits in order to match the total US benefit amount.

Before running the imputation procedure, one must do two things (besides downloading the appropriate CPS file into your directory):

1. Open and run the _Rf\_probs.py_ script. 
- This creates the Random Forest Classifer probabilities used in the imputation labeled as _rf\_probs.csv_. 

2. Open and run the _arma.ipynb_ script after downloading table 1 in the documentation pdf and entitling it _claim\_totals.csv_, or just use the claims_predicted file in the repository.
- This will create the claims_predicted file used in the imputation.

## Imputing Recipients

We create other variables that correspond to WC eligibility rules, as described in our documentation, to increase the predictive power of our logistic regression.

We then use the following logistic regression to determine the likelihood of an individual
participating in the program:

_WC\_YN_ = &alpha; + &beta; _Armed Forces_ + &beta; _Construction_ +
                        &beta; _Educational and Health Services_ + &beta; _Financial Activities_ +
                        &beta; _Information_ + &beta; _Leisure and hospitality_ +
			&beta; _Manufacturing_ + &beta; _Mining_ +
                        &beta; _Other services_ + &beta; _Professional and business services_ +
			&beta; _Public administration_ + &beta; _Transportation and utilities_ +
                        &beta; _Wholesale and retail trade_ + &beta; _age\_squared_ +
			&beta; _dis\_cs_ + &beta; _dis\_hp_ + 
			&beta; _finc\_dis_ + &beta; _cov\_hi_ + &beta; _gender_ +&epsilon;


Individuals are then ranked by the probability of participation. Then we aggregate all the weights of those who are current WC recipients, and add on the most likely participants who aren't currently receiving WC, until we reach the administrative
level totals for the US.

In Rf_probs.py, we use a more accurate Random Forest Classifier (RFC) model, instead of a logistic regression, to determine the
likelihood of individual WC participation. We then repeat the same step above with the new probabilities.

## Benefit Imputation

For each imputed recipient, we assign the average benefit amount. After imputation, We then calculate the total benefits for all recipients and compare this total to SSA/NASI administrative data. The adjustment ratio is then determined by
dividing total administrative benefits by total imputed benefits.

### Continuous Updating
The following variables will need to be updated in the model and this document
when moving to the next year:
1. CPS, should be for the year after the year being considered (since WC is reported for the previous year)
2. Hard-coded administrative totals in _WC\_impute.py_ labeled _total\_covered_, _total\_benefits_, and _claims\_projected_ should be changed 
according to values given in the NASI annual report. The values for these variables can be found on Table 1, and 18 respectively. If the claims per 100,000 insured total is is already given in the NASI report for the year considered, then do not project using our arma.ipynb model. Instead, hard code in the claim number for that year. Otherwise, update the _claim\_totals.csv_  file with the latest claim totals from table 18 and project the claim total up to the most recent year needed using arma.ipynb.
