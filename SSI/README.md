# Supplemental Security Income Imputation

The Social Security Administration (SSA) reported 9.3 million SSI participants in 2014, amounting to $55 billion in federal benefits and $4.6 billion in state-administered benefits.
The raw 2014 CPS reports just 6.1 million participants receiving $47.1 billion in benefits. Accordingly, we estimate that SSI participation is under reported by at most 3.1 million and SSI benefits are underreported by, at most, $12.5 billion.

## CPS Micro-Data and SSA Targets
In the SSI section of the CPS March Supplement, each respondent marked as a recipient of SSI also reports the total benefit they and their child receives.

The targets for the imputation are primarily based on the SSA annual report.
In addition to the aggregates, we target each state's participants based on age group and dollar benefit subtotals.
There are also several adjustments made before imputation.

First, SSI is administered at both the state and federal level.
We assume an average SSI recipient in the CPS does not distinguish between the source of their SSI check, therefore we want to include the state counterpart.
However, statistics for most state administered programs are not available after 2011.
In these cases, we take the last year available from the SSA and extrapolate each state's total benefits by the growth rate of federal payments for individual recipients.

Second, the CPS covers the non-institutionalized population in the US, but SSI also pays a portion of room and board for people living in nursing homes and other assisted living facilities, which are likely not included in the CPS sample.
State subtotals by living arrangements are not available, so we relied on the ratio between total benefits received by non-institutionalized and institutionalized individuals at the federal level.
We found that about 99.79% of benefits go to the non-institutionalized, thus we decided to ignore the institutionalized portion of the benefits at both the state and federal levels.

Third, subtotals provided by the SSA, in terms of recipients and benefit amounts, are only available for December 2014.
For each age group at the state level, we amplify the number of recipients by the ratio of full-year to December payments across the US.

## Imputation Procedure

Recipients and benefits were imputed using a two step procedure, first adding individuals to the recipients pool to match administrative totals, and then imputing and adjusting benefits to match the dollar total.

### Imputing Recipients

We use a logistic regression to estimate the likelihood of an individual being a recipient.
Our model is based on SSI program rules and the independent variables include countable income, age, and disability status.

y = &alpha; + &beta; countable income + &beta; age + &beta; disability + &epsilon;

Countable income includes earned and unearned income, and excludes the first $20 of most income received, and $65 of earnings over one-half of earning above $65. Age is an indicator for whether they are older than 65 or not, and disability is a combined indicator which covers both physical disability indicated in the CPS and work disability, as defined by the census.

### Imputing and adjusting Benefit Amounts

For each imputed recipient, we initially assign the average benefit in their state and age group. Aggregate benefits are still lower than the administrative total after this step, so we augment each record's benefit proportionally by the ratio of administrative total over current aggregates. 
