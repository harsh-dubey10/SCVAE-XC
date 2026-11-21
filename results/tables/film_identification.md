# Cross-culture film identification (SEED-GER <-> SEED-FRA shared films)

Balanced accuracy (%), mean over test people; above = test people above chance; p from 2000 block permutations of the test films (one-sided).

## negative: Dead Man Walking, A Perfect World, Schindler's List (chance 33%)

| Train → Test | Modality | Accuracy | Above chance | p |
|---|---|---:|---:|---:|
| German → French | EEG + eye | 29.6 | 3/8 | 0.7386 |
| German → French | EEG | 20.2 | 1/8 | 0.9895 |
| German → French | eye | 13.4 | 0/8 | 1.0000 |
| French → German | EEG + eye | 40.1 | 4/8 | 0.1029 |
| French → German | EEG | 21.1 | 1/8 | 0.9790 |
| French → German | eye | 20.4 | 0/8 | 0.9845 |

## negative_2: Dead Man Walking, A Perfect World (chance 50%)

| Train → Test | Modality | Accuracy | Above chance | p |
|---|---|---:|---:|---:|
| German → French | EEG + eye | 65.0 | 5/8 | 0.0105 |
| German → French | EEG | 55.3 | 4/8 | 0.1204 |
| German → French | eye | 84.8 | 8/8 | 0.0005 |
| French → German | EEG + eye | 90.2 | 8/8 | 0.0005 |
| French → German | EEG | 42.7 | 2/8 | 0.8226 |
| French → German | eye | 95.8 | 8/8 | 0.0005 |

## positive: The Dinner Game, The Visitors (chance 50%)

| Train → Test | Modality | Accuracy | Above chance | p |
|---|---|---:|---:|---:|
| German → French | EEG + eye | 20.8 | 0/8 | 1.0000 |
| German → French | EEG | 20.7 | 0/8 | 1.0000 |
| German → French | eye | 45.0 | 2/8 | 0.7771 |
| French → German | EEG + eye | 29.8 | 1/8 | 0.9925 |
| French → German | EEG | 29.7 | 0/8 | 0.9985 |
| French → German | eye | 51.7 | 2/8 | 0.3858 |

## all_5: Dead Man Walking, A Perfect World, Schindler's List, The Dinner Game, The Visitors (chance 20%)

| Train → Test | Modality | Accuracy | Above chance | p |
|---|---|---:|---:|---:|
| German → French | EEG + eye | 13.8 | 1/8 | 0.9530 |
| German → French | EEG | 13.4 | 1/8 | 0.9650 |
| German → French | eye | 9.3 | 0/8 | 0.9990 |
| French → German | EEG + eye | 19.5 | 2/8 | 0.5447 |
| French → German | EEG | 13.9 | 1/8 | 0.9465 |
| French → German | eye | 28.7 | 7/8 | 0.0220 |

## Timing control: fraction of each test-culture negative film called 'A Perfect World'

By a Dead Man Walking vs A Perfect World model trained on the other culture. Start minute in the test session in brackets.

| German → French | Dead Man Walking | A Perfect World | City of Angels (4) | E.T. (11) | Dangerous Mind (46) | Life is Beautiful (56) |
|---|---:|---:|---:|---:|---:|---:|
| EEG + eye | 0.27 | 0.57 | 0.11 | 0.23 | 0.18 | 0.33 |
| EEG | 0.16 | 0.26 | 0.07 | 0.19 | 0.31 | 0.53 |
| eye | 0.16 | 0.85 | 0.24 | 0.26 | 0.08 | 0.21 |

| French → German | Dead Man Walking | A Perfect World | The Piano (24) | Saving Private Ryan (42) | Misery (55) |
|---|---:|---:|---:|---:|---:|
| EEG + eye | 0.05 | 0.81 | 0.91 | 0.10 | 0.62 |
| EEG | 0.76 | 0.60 | 0.72 | 0.41 | 0.56 |
| eye | 0.05 | 0.95 | 0.90 | 0.10 | 0.51 |
