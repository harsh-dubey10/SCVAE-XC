# Films-per-emotion learning curve

Two training cultures → the third; all source people, only the number of distinct source films per emotion varies. Logistic regression (C = 0.01). Accuracy %, mean over test people, mean ± sd over 10 random film draws. Design: docs/superpowers/specs/2026-10-29-film-learning-curve-design.md.

## eeg_eye

| Target | F = 1 | F = 2 | F = 3 | F = 4 | F = 6 | F = 8 | F = 9 | F = 10 | spread (budget of F = 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chinese | 54.2 ± 9.5 | 57.6 ± 10.6 | 61.5 ± 4.0 | 65.2 ± 5.6 | 67.7 ± 4.0 | 66.8 ± 3.8 | – | 66.4 ± 1.4 | 68.3 ± 3.4 |
| German | 48.8 ± 5.5 | 53.6 ± 5.0 | 52.9 ± 6.5 | 56.4 ± 4.3 | 55.3 ± 3.2 | 57.5 ± 1.4 | 57.7 ± 1.4 | – | 58.2 ± 1.5 |
| French | 47.2 ± 5.9 | 55.0 ± 5.6 | 56.5 ± 2.9 | 55.7 ± 2.9 | 59.7 ± 2.1 | 60.2 ± 1.2 | – | – | 61.5 ± 1.1 |
| **Mean** | **50.1** | **55.4** | **57.0** | **59.1** | **60.9** | **61.5** | – | – | **62.6** |

- F = 8 − F = 4: +2.4 points; spread − concentrated (F = 2 budget): +7.2; fitted slope +3.8 points per doubling of F.

**Pre-registered decision (EEG+eye): MORE FILMS HELP.**

## eeg

| Target | F = 1 | F = 2 | F = 3 | F = 4 | F = 6 | F = 8 | F = 9 | F = 10 | spread (budget of F = 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chinese | 51.8 ± 9.2 | 54.3 ± 6.4 | 56.6 ± 4.1 | 60.4 ± 3.9 | 62.2 ± 2.1 | 60.4 ± 2.4 | – | 61.2 ± 0.8 | 62.7 ± 2.1 |
| German | 43.9 ± 4.6 | 47.5 ± 5.0 | 48.0 ± 4.9 | 48.2 ± 4.6 | 48.4 ± 2.6 | 50.4 ± 2.3 | 49.5 ± 2.0 | – | 51.4 ± 1.7 |
| French | 45.3 ± 4.7 | 50.1 ± 5.0 | 50.9 ± 2.2 | 50.5 ± 3.8 | 53.9 ± 2.8 | 53.2 ± 1.0 | – | – | 54.5 ± 1.1 |
| **Mean** | **47.0** | **50.6** | **51.8** | **53.0** | **54.8** | **54.6** | – | – | **56.2** |

- F = 8 − F = 4: +1.6 points; spread − concentrated (F = 2 budget): +5.6; fitted slope +2.6 points per doubling of F.

