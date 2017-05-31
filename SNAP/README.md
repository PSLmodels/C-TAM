# Supplemental Nutrition Assistance Program Imputation

SNAP, previously known as Food Stamps,is reported on a household, rather than
individual, level in the CPS. Therefore, each member of the household will report
the same benefit amounts and time receiving benefits. For example, if one
person in the household reports receiving $1,000 in benefits for four months,
everyone in the household will report the same.

Target data for the imputation came from official [SNAP data](https://www.fns.usda.gov/sites/default/files/ops/Characteristics2014.pdf).
In Fiscal Year 2014, an average of 22.74 million households claimed about
$70 billion in benefits.

## Imputation Procedure

Recipients and benefits are imputed using a two step procedure. First, participation
is imputed to match administrative totals. Second, we impute an adjustment ratio
for benefits in order to match total benefit amounts.

## Imputing Recipients

We initially calculate net income according to program rules:

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
| [_Source_](https://www.fns.usda.gov/snap/cost-living-adjustment-cola-information)| |

4. For households with members ages 60 or older, medical expenses exceeding $35,
if they are not paid by insurance or someone else, are also deducted
5. Child support can be deducted
6. Dependent care and shelter costs may also be deducted. We use the average
official numbers of $10 and $290 a month. These costs are not provided by the
CPS, so we instead use numbers from the USDA.

We then use the following regression to determine the likelihood of a household
participating in the program:

_indicator_ = &alpha; + &beta; _net income_ + &beta; _household size_ +
                        &beta; _disability_ + &beta; _number of children_ +
                        &beta; _ABAWD_ + &beta; _welfare participation_ + &epsilon;

Able-Bodied Adults Without Dependents (ABAWD) are not allowed to stay on SNAP
for more than three months, thus their households have a lower chance of receiving
benefits.

Disability and welfare participation are included to indicate any disabilities
in the household and participation in other welfare programs, respectively.

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
compare imputed benefits to SNAP administrative data. The accuracy of our imputations
varies state-by-state, so adjustment ratios are determined for each state by
dividing total administrative benefits by total imputed benefits. Most ratios
are close to one and are used to either increase or shrink each household's
benefits.

### Continuous Updating
The following variables will need to be updated in the model and this document
when moving to the next year:
1. Deduction amounts based on household size
2. Average dependent care and shelter costs
3. SNAP income eligibility standards
4. Total benefit and participation levels
5. State-level targets
