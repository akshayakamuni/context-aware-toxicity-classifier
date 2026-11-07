# Label-Wise Conversation Context Audit

## 1. Purpose

This audit examines whether conversation context availability and reconstructable ancestor depth differ across the five official CAD v1.1 labels.

Only same-split ancestors with non-empty text are counted.

## 2. Validation

- Official records audited: 23,417
- Unique official IDs: 23,417
- Records with at least one ancestor: 12,835
- Records with at least two ancestors: 7,005
- Records with at least three ancestors: 3,188

## 3. Label-wise summary

| label                    |   records |   percentage_of_all_official_records |   mean_ancestor_depth |   median_ancestor_depth |   records_with_parent_context |   parent_context_percentage |   records_with_two_ancestors |   two_ancestor_percentage |   records_with_three_ancestors |   three_ancestor_percentage |
|:-------------------------|----------:|-------------------------------------:|----------------------:|------------------------:|------------------------------:|----------------------------:|-----------------------------:|--------------------------:|-------------------------------:|----------------------------:|
| Neutral                  |     19107 |                               81.595 |                1.0866 |                       1 |                         10418 |                      54.525 |                         5683 |                    29.743 |                           2588 |                      13.545 |
| IdentityDirectedAbuse    |      2305 |                                9.843 |                1.0399 |                       1 |                          1250 |                      54.23  |                          663 |                    28.764 |                            292 |                      12.668 |
| AffiliationDirectedAbuse |      1199 |                                5.12  |                1.0517 |                       1 |                           670 |                      55.88  |                          351 |                    29.274 |                            150 |                      12.51  |
| PersonDirectedAbuse      |      1054 |                                4.501 |                1.3083 |                       1 |                           640 |                      60.721 |                          382 |                    36.243 |                            198 |                      18.786 |
| CounterSpeech            |       217 |                                0.927 |                1.5806 |                       1 |                           149 |                      68.664 |                          106 |                    48.848 |                             51 |                      23.502 |

## 4. Label-wise depth distribution

| label                    |   depth |   records |   percentage_within_label |
|:-------------------------|--------:|----------:|--------------------------:|
| Neutral                  |       0 |      8689 |                    45.475 |
| Neutral                  |       1 |      4735 |                    24.781 |
| Neutral                  |       2 |      3095 |                    16.198 |
| Neutral                  |       3 |      1397 |                     7.311 |
| Neutral                  |       4 |       663 |                     3.47  |
| Neutral                  |       5 |       403 |                     2.109 |
| Neutral                  |       6 |        42 |                     0.22  |
| Neutral                  |       7 |        24 |                     0.126 |
| Neutral                  |       8 |        21 |                     0.11  |
| Neutral                  |       9 |        14 |                     0.073 |
| Neutral                  |      10 |        10 |                     0.052 |
| Neutral                  |      11 |         7 |                     0.037 |
| Neutral                  |      12 |         4 |                     0.021 |
| Neutral                  |      13 |         3 |                     0.016 |
| IdentityDirectedAbuse    |       0 |      1055 |                    45.77  |
| IdentityDirectedAbuse    |       1 |       587 |                    25.466 |
| IdentityDirectedAbuse    |       2 |       371 |                    16.095 |
| IdentityDirectedAbuse    |       3 |       166 |                     7.202 |
| IdentityDirectedAbuse    |       4 |        71 |                     3.08  |
| IdentityDirectedAbuse    |       5 |        51 |                     2.213 |
| IdentityDirectedAbuse    |       6 |         3 |                     0.13  |
| IdentityDirectedAbuse    |      13 |         1 |                     0.043 |
| AffiliationDirectedAbuse |       0 |       529 |                    44.12  |
| AffiliationDirectedAbuse |       1 |       319 |                    26.606 |
| AffiliationDirectedAbuse |       2 |       201 |                    16.764 |
| AffiliationDirectedAbuse |       3 |        92 |                     7.673 |
| AffiliationDirectedAbuse |       4 |        29 |                     2.419 |
| AffiliationDirectedAbuse |       5 |        28 |                     2.335 |
| AffiliationDirectedAbuse |       8 |         1 |                     0.083 |
| PersonDirectedAbuse      |       0 |       414 |                    39.279 |
| PersonDirectedAbuse      |       1 |       258 |                    24.478 |
| PersonDirectedAbuse      |       2 |       184 |                    17.457 |
| PersonDirectedAbuse      |       3 |       103 |                     9.772 |
| PersonDirectedAbuse      |       4 |        48 |                     4.554 |
| PersonDirectedAbuse      |       5 |        40 |                     3.795 |
| PersonDirectedAbuse      |       6 |         2 |                     0.19  |
| PersonDirectedAbuse      |       7 |         3 |                     0.285 |
| PersonDirectedAbuse      |       8 |         1 |                     0.095 |
| PersonDirectedAbuse      |      11 |         1 |                     0.095 |
| CounterSpeech            |       0 |        68 |                    31.336 |
| CounterSpeech            |       1 |        43 |                    19.816 |
| CounterSpeech            |       2 |        55 |                    25.346 |
| CounterSpeech            |       3 |        27 |                    12.442 |
| CounterSpeech            |       4 |        13 |                     5.991 |
| CounterSpeech            |       5 |         9 |                     4.147 |
| CounterSpeech            |       6 |         2 |                     0.922 |

## 5. Context annotation distribution

| label                    | context_annotation   |   records |   percentage_within_label |
|:-------------------------|:---------------------|----------:|--------------------------:|
| Neutral                  | NoAnnotation         |     18999 |                    99.435 |
| Neutral                  | CurrentOnly          |       108 |                     0.565 |
| IdentityDirectedAbuse    | CurrentOnly          |      1599 |                    69.371 |
| IdentityDirectedAbuse    | PreviousOnly         |       603 |                    26.161 |
| IdentityDirectedAbuse    | CurrentAndPrevious   |       103 |                     4.469 |
| AffiliationDirectedAbuse | CurrentOnly          |       860 |                    71.726 |
| AffiliationDirectedAbuse | PreviousOnly         |       291 |                    24.27  |
| AffiliationDirectedAbuse | CurrentAndPrevious   |        48 |                     4.003 |
| PersonDirectedAbuse      | CurrentOnly          |       688 |                    65.275 |
| PersonDirectedAbuse      | PreviousOnly         |       286 |                    27.135 |
| PersonDirectedAbuse      | CurrentAndPrevious   |        80 |                     7.59  |
| CounterSpeech            | PreviousOnly         |       194 |                    89.401 |
| CounterSpeech            | CurrentAndPrevious   |        23 |                    10.599 |

## 6. Chain termination reasons

| split   | termination_reason                   |   records |
|:--------|:-------------------------------------|----------:|
| dev     | parent_missing_from_metadata         |      4376 |
| dev     | parent_excluded_from_official_splits |       150 |
| test    | parent_missing_from_metadata         |      5172 |
| test    | parent_excluded_from_official_splits |       135 |
| train   | parent_missing_from_metadata         |     13144 |
| train   | parent_excluded_from_official_splits |       439 |
| train   | parent_text_missing                  |         1 |

## 7. Interpretation

- Compare parent-context availability across labels.
- Compare average and median conversation depth.
- Check whether rare labels have less available context.
- Context annotation is not equivalent to actual parent-text availability.
- Multi-label records contribute to each label they carry.
- These descriptive results do not establish causation.

## 8. Research decisions

- Preserve records with missing context.
- Do not fabricate missing conversation text.
- Use only same-split ancestor text.
- Never use ancestor labels as model inputs.
- Evaluate context-available and context-unavailable subsets separately.
- Use identical official splits for all context experiments.