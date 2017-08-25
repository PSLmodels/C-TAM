# WIC Program Imputation

Women, Infants, and Children (WIC) is reported on an individual level in the CPS.

Target data for the imputation came from official [USDA data](https://www.fns.usda.gov/pd/wic-program),
called "WICAgencies2014ytd.xls", from the .xls link on "FY 2014 (final)". 

We use the 2015 CPS because the WIC variable, WICYN, reports the respondents' WIC participation recipiency for the calendar year 2014, rather than 2015.


The Special Supplemental Nutrition Program for Women, Infants, and Children (WIC) is a federal
assistance program intended to provide a safety net for low-income families by providing nutritional
supplementation. Low-income women who have at risk pregnancies, or who are nutritionally deficient
while breastfeeding may qualify for WIC supplementation, as do low-income, at risk infants (age 0),
and children (age 1 up to age 5).


In fiscal year 2014, according to _"WICAgencies2014ytd.xls"_,
approximately 8 million individuals claimed roughly 4.1 billion dollars in WIC benefits. The annual average combined benefit for each recipient is approximately $512 with significant variation across states.

Correspondingly, 2015 CPS totals, "WICYN", indicate that around 4.2 million individuals received WIC benefits, with no information regarding benefit amounts; 
however, this number does not accurately capture the distribution of infant, child,
and women recipients. According to a report by Suzanne Macartney at the US Census Bureau, the
CPS WICYN variable falls short by failing to capture any infant or child WIC recipients, while
overreporting women WIC recipients. This is shown using the following USDA administrative data.
According to the 2014 USDA administrative data on the WIC program, there were 1.9 million infant, 4.2
million child, and 1.9 million woman WIC participants; on the contrary, according to the 2015 CPS
WICYN variable (which reports 2014 WIC participation), there were only 4,409 child (ages 0-15),
and 4.2 million woman weighted WIC participants. Moreover, this relatively small number of
children could be attributed to editing errors. Since CPS WIC participants are almost entirely women
over the age of 15, Macartney concluded that WICYN is actually reporting the number of families
participating in WIC, rather than individual level participation. More specifically, WICYN is instead
showing the number of women who have children who are, and/or who themselves are, receiving
WIC benefits. 

To account for this discrepancy we used a preliminary imputation method, which followed
Macartney's method for imputing infant and child participation closely, as well as reducing the
amount of woman participants; this method is described in our documentation pdf entitled _WIC\_Imputation\_Report.pdf_ above. 


## Imputation Procedure

After our preliminary imputation, recipients and benefits are imputed using a two step procedure. First, participation
is imputed to match administrative totals through an augmentation process.
Second, we impute an adjustment ratio for benefits in order to match total benefit amounts.

Before running the imputation procedure, one must do two things (besides downloading the appropriate CPS file into your directory):

1. Open and run the _Rf\_probs.py_ script. 
- This creates the Random Forest Classifer probabilities used in the imputation labeled as _rf\_probs\_women.csv_, _rf\_probs\_infants.csv_, and _rf\_probs\_children.csv_. 

2. Run _create\_admin.py_ with the appropriate [USDA data](https://www.fns.usda.gov/pd/wic-program) already downloaded into the imputation script location.  (refer to documentation for more information).
- Download _"WICAgencies2014ytd.xls"_. This creates the administrative data in the file _Admin\_totals\_all.csv_ used in the imputation.

## Imputing Recipients

Since WIC program rules require earnings to be below a threshold, we use the _fwsval_ variable as an income proxy. We then create a binary variable called _income\_eligibility_, signifying all those families who earn below eligibility guidelines established by [USDA](https://www.fns.usda.gov/wic/wic-income-eligibility-guidelines-2012-2013) for the year 2013-2014, as a proxy for those who are eligible based on income rules.

We create other variables that correspond to WIC eligibility rules, as described in our documentation, to increase the predictive power of our logistic regression.

We then use the following logistic regression to determine the likelihood of an individual
participating in the program, for three different categories: women, infants, and children:

_WIC\_infant_ = &alpha; + &beta; _hfdval_ + &beta; _cov\_hi_ +
                        &beta; _ch\_mc_ + &beta; _infant_ +
                        &beta; _fwsval_ + &epsilon;
			

_WIC\_child_ = &alpha; + &beta; _hfdval_ + &beta; _cov\_hi_ +
                        &beta; _ch\_mc_ + &beta; _child_ +
                        &beta; _fwsval_ + &epsilon;

_WIC\_woman_ = &alpha; + &beta; _hfdval_ + &beta; _rsnnotw_ +
                        &beta; _has\_child_ + &beta; _woman_ +
                        &beta; _fwsval_ + beta; _income\_eligibility_ + &beta; _caid_ +
			&epsilon;


After, we use the fitted coefficients to produce a vector of
probabilities for WIC recipients in the three categories. We then rank all recipients within each
category according to their fitted probability. For each state sub-group, we aggregate the recipient
weights, and add extra non-recipients by likelihood until the weights reach administrative level, for
each category.

In Rf_probs.py, we use a more accurate Random Forest Classifier (RFC) model, instead of a logistic regression, to determine the
likelihood of individual WIC participation. We then repeat the same step above with the new RFC probabilities.

## Benefit Imputation

For each imputed/augmented recipient, we assign the average benefit amount for the corresponding
state, just like we did for all other recipients. We then calculate the new total outlays for each state,
and compare these outlays with USDA administrative state outlays. We calculate the adjustment
ratios for each state by dividing administrative outlays by the new outlays. Most adjustment ratios
close to 1, but some are significantly larger. We use these adjustment ratios to augment individual's
benefits to match the state administrative totals.

### Continuous Updating
The following variables will need to be updated in the model and this document
when moving to the next year:
1. CPS, should be for the year after the year being considered (since WIC participation is reported for the previous year in CPS). Update all the CPS file names to the year considered.
2. WIC income eligibility standards for the year considered. Change the variables "base" and "step" in WIC\_impute.py, under the _income\_lim\_indator_ function, according to these new standards. You can find these eligibility guidelines on USDA website.
3. In _create\_admin.py_, modify the year used in the code to the year considered, and download the appropriate data for that year.
