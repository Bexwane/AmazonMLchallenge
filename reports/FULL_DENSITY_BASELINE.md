# Full-Density Blocking Baseline (all train Source-1, no sampling)

Blocker: `E003-style k_comb=45 k_name=10 df_cap=2500`

| metric | value |
|---|---|
| pairs | 104293506 |
| pairs_per_s1 | 47.2596 |
| pair_recall | 0.8610 |
| recall_India | 0.7933 |
| recall_US | 0.9063 |
| recall_S2 | 0.8561 |
| recall_S3 | 0.8656 |
| recall@1 | 0.2448 |
| recall@3 | 0.5836 |
| recall@5 | 0.7292 |
| recall@10 | 0.7986 |
| recall@20 | 0.8292 |
| recall@30 | 0.8429 |
| oracle_macro_f05 | 0.9338 |
| blocking runtime (min) | 11.7 |
| total runtime incl. diagnosis (min) | 17.6 |
| peak RAM after blocking | 21.2 GiB (children 0.21) |
| peak RAM total | 21.2 GiB (children 0.21) |

## Present vs absent

| bucket | true pairs | share | share_India | share_US |
|---|---|---|---|---|
| absent from candidate set | 1,061,608 | 0.1390 | 0.2067 | 0.0937 |
| present, rank 1-10 | 6,099,840 | 0.7986 | 0.7213 | 0.8502 |
| present, rank 11-20 | 234,082 | 0.0306 | 0.0311 | 0.0304 |
| present, rank 21-30 | 104,795 | 0.0137 | 0.0157 | 0.0124 |
| present, rank > 30 | 138,040 | 0.0181 | 0.0253 | 0.0132 |
| present but outside top-10 | 476,917 | 0.0624 | 0.0720 | 0.0560 |
| present but outside top-20 | 242,835 | 0.0318 | 0.0410 | 0.0257 |
| present but outside top-30 | 138,040 | 0.0181 | 0.0253 | 0.0132 |

Absent pairs with no surviving key (name_cos + addr_cos = 0): 18.14%.
Full-pool rank quantiles of scored-but-absent pairs (sample 11,303): {'0.1': 2.0, '0.25': 49.0, '0.5': 84.0, '0.75': 257.0, '0.9': 747.0}