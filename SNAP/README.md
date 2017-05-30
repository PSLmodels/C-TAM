# Supplemental Nutrition Assistance Program Imputation

SNAP, previously known as Food Stamps,is reported on a household, rather than
individual, level in the CPS. Therefore, each member of the household will report
the same benefit amounts and time receiving benefits. For example, if one
person in the household reports receiving $1,000 in benefits for four months,
everyone in the household will report the same.

Target data for the imputation came from official SNAP data. In Fiscal Year 2014,
an average of 22.74 million households claimed about $70 billion in benefits.

## Imputation Procedure

Recipients and benefits are imputed using a two step procedure. First, participation
is imputed to match administrative totals. Second, we impute an adjustment ratio
for benefits in order to match total benefit amounts.

## Imputing Recipients

A basic linear regression is used to predict the likelihood of participation in
the SNAP program. In compliance with SNAP rules, only net income is used as an
independent variable. Net income is computed using the following rules, also in
compliance with program rules:
1. Gross income for each household is attained by summing up the earned and
unearned income of all household members.
2. A deduction of 20% of earned income is applied.
3. An additional deduction is applied according to household size

|Size | Deduction|
|------|----------|
| <= 3| $155 |
| 4 | $168 |
| 5  | $197 |
| >= 6| $ 226|

4. For households with members ages 60 or older, medical expenses exceeding $35,
if they are not paid by insurance or someone else, are also deducted
5. Child support can be deducted
6. Dependent care and shelter costs may also be deducted. We use the average
official numbers of $10 and $290 a month. These costs are not provided by the
CPS, so we instead use numbers from the USDA.

The final model is:

_participation_ = &alpha; + &beta; _net income_ + &epsilon;

SNAP rules set upper bounds on monthly income in order to be allowed to participate
in the program, the highest of which is $5,490. Therefore, we only run households
with monthly income bellow this level through the model.

Households are then ranked by the probability of participation, and, for each
state's subgroup, their weights are summed until they reach the administrative
level. According to CPS and USDA, the gap between individual and household
recipients reported to the CPS and administrative totals is 10.5 million and 10.3
million, respectively. Because of this, the size of each household is limited
to one person when ranking participation probability. This is the only way both
individual and household numbers can match administrative totals.

## Benefit Imputation

For each imputed household recipient, we assign the average benefit amount for
their state. We then calculate the total dollar benefits for each state and
compare imputed benefits to SNAP administrative data. Exact qualifications
for SNAP eligibility vary state by state, so adjustment ratios are determined
for each state by dividing total administrative benefits by total imputed benefits.
Most ratios are close to one and are used to either increase or shrink each
household's benefits.
