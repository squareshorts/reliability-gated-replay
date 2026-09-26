# Cross-Tranche Diagnostic Report

## 80-Run Verification
- total_expected: 80
- total_found: 80
- unique_scientific_runs: 80
- tranches_have_20: True
- seeds_per_combo: True
- has_nan_acc: False
- matched_cap_per_tranche: False

## Method Summaries
|                                         |   ('final_avg_acc', 'mean') |   ('final_avg_acc', 'std') |   ('buffer_purity', 'mean') |   ('buffer_purity', 'std') |   ('normalized_class_entropy', 'mean') |   ('normalized_class_entropy', 'std') |   ('class_count_variance', 'mean') |   ('class_count_variance', 'std') |   ('minority_class_coverage', 'mean') |   ('minority_class_coverage', 'std') |   ('replay_loss_clean', 'mean') |   ('replay_loss_clean', 'std') |   ('replay_loss_noisy', 'mean') |   ('replay_loss_noisy', 'std') |
|:----------------------------------------|----------------------------:|---------------------------:|----------------------------:|---------------------------:|---------------------------------------:|--------------------------------------:|-----------------------------------:|----------------------------------:|--------------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------:|--------------------------------:|-------------------------------:|
| ('seq_cifar10', 'sym20', 'er')          |                     0.19984 |                 0.0200167  |                      0.7932 |                 0.0280214  |                               0.994369 |                           0.00240598  |                              64.12 |                          27.3571  |                                     1 |                                    0 |                       1.59042   |                      1.54457   |                       1.25644   |                      1.24103   |
| ('seq_cifar10', 'sym20', 'gate_conf')   |                     0.21944 |                 0.016363   |                      0.8124 |                 0.00712741 |                               0.992867 |                           0.00184409  |                              80.8  |                          20.2064  |                                     1 |                                    0 |                       0.778165  |                      0.417415  |                       0.332869  |                      0.545392  |
| ('seq_cifar10', 'sym20', 'gate_loss')   |                     0.23808 |                 0.0283565  |                      0.962  |                 0.00447214 |                               0.980613 |                           0.00259343  |                             234.04 |                          31.9285  |                                     1 |                                    0 |                       4.24277   |                      2.08113   |                       0.954091  |                      0.656588  |
| ('seq_cifar10', 'sym20', 'oracle')      |                     0.22124 |                 0.0183343  |                      1      |                 0          |                               0.997854 |                           0.00107053  |                              24.72 |                          12.2814  |                                     1 |                                    0 |                       1.94216   |                      0.456163  |                     nan         |                    nan         |
| ('seq_cifar10', 'sym60', 'er')          |                     0.13984 |                 0.00807019 |                      0.4044 |                 0.0331482  |                               0.996418 |                           0.000750229 |                              41.36 |                           8.58417 |                                     1 |                                    0 |                       0.190866  |                      0.290474  |                       0.0756064 |                      0.0626515 |
| ('seq_cifar10', 'sym60', 'gate_conf')   |                     0.1442  |                 0.00468402 |                      0.4052 |                 0.0163463  |                               0.991245 |                           0.00417862  |                              98.96 |                          45.7911  |                                     1 |                                    0 |                       0.0441251 |                      0.024088  |                       0.0412429 |                      0.0239411 |
| ('seq_cifar10', 'sym60', 'gate_loss')   |                     0.17048 |                 0.021446   |                      0.5476 |                 0.0632835  |                               0.912352 |                           0.0813623   |                            1148    |                        1061.99    |                                     1 |                                    0 |                       1.13708   |                      1.00857   |                       0.659643  |                      0.648427  |
| ('seq_cifar10', 'sym60', 'oracle')      |                     0.20116 |                 0.0218671  |                      1      |                 0          |                               0.994418 |                           0.00153974  |                              64.4  |                          18.4591  |                                     1 |                                    0 |                       3.57453   |                      2.90904   |                     nan         |                    nan         |
| ('split_cifar10', 'sym20', 'er')        |                     0.663   |                 0.0232932  |                      0.7932 |                 0.0280214  |                               0.994811 |                           0.00142211  |                              58.92 |                          15.6369  |                                     1 |                                    0 |                       0.0616041 |                      0.0363442 |                       0.127217  |                      0.0942428 |
| ('split_cifar10', 'sym20', 'gate_conf') |                     0.67086 |                 0.016524   |                      0.8376 |                 0.019718   |                               0.980335 |                           0.00481594  |                             223.4  |                          58.2507  |                                     1 |                                    0 |                       0.0436072 |                      0.0109391 |                       0.176697  |                      0.0792775 |
| ('split_cifar10', 'sym20', 'gate_loss') |                     0.70774 |                 0.018612   |                      0.8612 |                 0.0168879  |                               0.995855 |                           0.00117149  |                              46.92 |                          13.5496  |                                     1 |                                    0 |                       0.0434431 |                      0.0247233 |                       0.0797526 |                      0.0465089 |
| ('split_cifar10', 'sym20', 'oracle')    |                     0.75922 |                 0.0122334  |                      1      |                 0          |                               0.997854 |                           0.00107053  |                              24.72 |                          12.2814  |                                     1 |                                    0 |                       0.0480966 |                      0.0170928 |                     nan         |                    nan         |
| ('split_cifar10', 'sym60', 'er')        |                     0.44696 |                 0.0297799  |                      0.4044 |                 0.0331482  |                               0.995503 |                           0.00160986  |                              50.48 |                          16.9644  |                                     1 |                                    0 |                       0.114759  |                      0.0250082 |                       0.0824793 |                      0.024702  |
| ('split_cifar10', 'sym60', 'gate_conf') |                     0.46122 |                 0.0102392  |                      0.3796 |                 0.0214196  |                               0.985724 |                           0.00370399  |                             164.52 |                          40.5077  |                                     1 |                                    0 |                       0.122217  |                      0.0799237 |                       0.164677  |                      0.169434  |
| ('split_cifar10', 'sym60', 'gate_loss') |                     0.44106 |                 0.0154709  |                      0.3756 |                 0.0129151  |                               0.993238 |                           0.00302272  |                              78.4  |                          35.6042  |                                     1 |                                    0 |                       0.0965653 |                      0.0399269 |                       0.0895012 |                      0.0411333 |
| ('split_cifar10', 'sym60', 'oracle')    |                     0.64256 |                 0.0298746  |                      1      |                 0          |                               0.994418 |                           0.00153974  |                              64.4  |                          18.4591  |                                     1 |                                    0 |                       0.0801897 |                      0.051614  |                     nan         |                    nan         |

## Cross-Condition Changes (sym60 - sym20)
### split_cifar10
#### er
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |  -0.21604     |
| buffer_purity            |  -0.3888      |
| normalized_class_entropy |   0.000692238 |
| class_count_variance     |  -8.44        |
| minority_class_coverage  |   0           |
| replay_loss_clean        |   0.0531546   |
| replay_loss_noisy        |  -0.0447374   |

#### gate_loss
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |   -0.26668    |
| buffer_purity            |   -0.4856     |
| normalized_class_entropy |   -0.0026178  |
| class_count_variance     |   31.48       |
| minority_class_coverage  |    0          |
| replay_loss_clean        |    0.0531222  |
| replay_loss_noisy        |    0.00974863 |

#### gate_conf
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |   -0.20964    |
| buffer_purity            |   -0.458      |
| normalized_class_entropy |    0.00538877 |
| class_count_variance     |  -58.88       |
| minority_class_coverage  |    0          |
| replay_loss_clean        |    0.0786096  |
| replay_loss_noisy        |   -0.0120202  |

#### oracle
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |   -0.11666    |
| buffer_purity            |    0          |
| normalized_class_entropy |   -0.00343656 |
| class_count_variance     |   39.68       |
| minority_class_coverage  |    0          |
| replay_loss_clean        |    0.0320931  |
| replay_loss_noisy        |  nan          |

### seq_cifar10
#### er
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |    -0.06      |
| buffer_purity            |    -0.3888    |
| normalized_class_entropy |     0.0020489 |
| class_count_variance     |   -22.76      |
| minority_class_coverage  |     0         |
| replay_loss_clean        |    -1.39955   |
| replay_loss_noisy        |    -1.18083   |

#### gate_loss
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |    -0.0676    |
| buffer_purity            |    -0.4144    |
| normalized_class_entropy |    -0.0682602 |
| class_count_variance     |   913.96      |
| minority_class_coverage  |     0         |
| replay_loss_clean        |    -3.10569   |
| replay_loss_noisy        |    -0.294447  |

#### gate_conf
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |   -0.07524    |
| buffer_purity            |   -0.4072     |
| normalized_class_entropy |   -0.00162146 |
| class_count_variance     |   18.16       |
| minority_class_coverage  |    0          |
| replay_loss_clean        |   -0.73404    |
| replay_loss_noisy        |   -0.291626   |

#### oracle
|                          |   mean_change |
|:-------------------------|--------------:|
| final_avg_acc            |   -0.02008    |
| buffer_purity            |    0          |
| normalized_class_entropy |   -0.00343656 |
| class_count_variance     |   39.68       |
| minority_class_coverage  |    0          |
| replay_loss_clean        |    1.63237    |
| replay_loss_noisy        |  nan          |
