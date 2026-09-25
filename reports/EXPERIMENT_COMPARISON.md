# Experiment Comparison (blocking only; no model CV during E004-E007)

Probe rows: sampled train Source-1 queries per country searched against the FULL country pool (exact full-density recall, sampling error about 0.3 pt; candidate counts scaled to all 2,206,821 train S1). Recall is measured on the deduplicated UNION; no RRF, no truncation.

Selection rule: a channel joins if it adds >= 0.002 recall and >= 0.0005 recall per extra candidate per S1, within 80 candidates per S1.

| Experiment | Spec | Candidates (est.) | Pairs/S1 | Pair Recall | India | US | Incremental Recall | CV F0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E003 (all S1, measured) | base | 104,293,506 | 47.3 | 0.8610 | 0.7933 | 0.9063 | baseline | not run |
| E003 baseline blocker | `base` | 104,337,811 | 47.3 | 0.8597 | 0.7915 | 0.9061 | +0.0000 | not run |
| E004 base + deterministic recovery | `base,conj:20,addr_only:5,rescue,name_noaddr:5` | 147,167,745 | 66.7 | 0.9773 | 0.9577 | 0.9906 | +0.1175 | not run |
| E005 base + BM25 | `base,bm25:30` | 141,717,147 | 64.2 | 0.9472 | 0.9134 | 0.9701 | +0.0874 | not run |
| E006 base + BGE-M3 targeted (native-script pool) | `base,bge_native:10` | 112,723,151 | 51.1 | 0.9031 | 0.8986 | 0.9061 | +0.0433 | not run |
| E006b base + e5-small targeted (ablation) | `base,e5_native:10` | 112,756,447 | 51.1 | 0.9023 | 0.8966 | 0.9061 | +0.0425 | not run |
| E007 best union (greedy over all useful channels) | `base,conj:20,bm25:5,bge_native:5,name_noaddr:5` | 145,368,556 | 65.9 | 0.9813 | 0.9681 | 0.9903 | +0.1216 | not run |
| ref: E003 score at K=200 (what 'more K' buys) | `base,base_wide:200` | 436,118,362 | 197.6 | 0.9340 | 0.9209 | 0.9429 | +0.0743 | not run |
| ref: union of every probed channel | `all` | 596,781,663 | 270.4 | 0.9911 | 0.9860 | 0.9946 | +0.1314 | not run |
| E007 (all S1, measured) | `base,conj:20,bm25:5,bge_native:5,name_noaddr:5` | 145,218,172 | 65.8 | 0.9818 | 0.9684 | 0.9908 | +0.1208 | not run |

## Greedy steps

**E004 base + deterministic recovery**

| add | recall | gain | extra pairs/S1 | recall per extra pair |
|---|---:|---:|---:|---:|
| conj:5 | 0.9450 | +0.0852 | 1.4 | 0.06119 |
| addr_only:5 | 0.9497 | +0.0047 | 0.8 | 0.00578 |
| conj:10 | 0.9668 | +0.0171 | 3.6 | 0.00471 |
| rescue | 0.9689 | +0.0021 | 2.1 | 0.00098 |
| name_noaddr:5 | 0.9714 | +0.0025 | 2.9 | 0.00085 |
| conj:20 | 0.9773 | +0.0059 | 8.5 | 0.00070 |

**E005 base + BM25**

| add | recall | gain | extra pairs/S1 | recall per extra pair |
|---|---:|---:|---:|---:|
| bm25:5 | 0.9155 | +0.0558 | 0.9 | 0.06010 |
| bm25:10 | 0.9329 | +0.0174 | 2.4 | 0.00718 |
| bm25:20 | 0.9426 | +0.0097 | 6.3 | 0.00153 |
| bm25:30 | 0.9472 | +0.0046 | 7.3 | 0.00063 |

**E006 base + BGE-M3 targeted (native-script pool)**

| add | recall | gain | extra pairs/S1 | recall per extra pair |
|---|---:|---:|---:|---:|
| bge_native:5 | 0.9009 | +0.0411 | 1.8 | 0.02248 |
| bge_native:10 | 0.9031 | +0.0022 | 2.0 | 0.00112 |

**E006b base + e5-small targeted (ablation)**

| add | recall | gain | extra pairs/S1 | recall per extra pair |
|---|---:|---:|---:|---:|
| e5_native:5 | 0.8998 | +0.0400 | 1.8 | 0.02175 |
| e5_native:10 | 0.9023 | +0.0025 | 2.0 | 0.00126 |

**E007 best union (greedy over all useful channels)**

| add | recall | gain | extra pairs/S1 | recall per extra pair |
|---|---:|---:|---:|---:|
| conj:5 | 0.9450 | +0.0852 | 1.4 | 0.06119 |
| bm25:5 | 0.9515 | +0.0066 | 0.6 | 0.01117 |
| bge_native:5 | 0.9614 | +0.0099 | 1.7 | 0.00592 |
| conj:10 | 0.9742 | +0.0128 | 3.6 | 0.00358 |
| name_noaddr:5 | 0.9765 | +0.0023 | 2.9 | 0.00079 |
| conj:20 | 0.9813 | +0.0048 | 8.5 | 0.00057 |

## Channels alone (per country)

| country | channel | recall | recall@10 | recall@45 | gain over base | new pairs/q | cand. increase % | sec (probe) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| India | base | 0.7915 | 0.7177 | 0.7841 | n/a | n/a | n/a | 97.0 |
| India | base_wide:200 | 0.9191 | 0.7180 | 0.7859 | 0.1293 | 150.8 | 318.3 | 3.0 |
| India | addr_only:20 | 0.6977 | 0.6564 | 0.6977 | 0.0909 | 8.2 | 17.3 | 2.0 |
| India | name_noaddr:10 | 0.0349 | 0.0349 | 0.0349 | 0.0066 | 6.0 | 12.6 | 1.1 |
| India | conj:50 | 0.9378 | 0.8893 | 0.9358 | 0.1675 | 42.6 | 90.0 | 146.2 |
| India | rescue | 0.5378 | 0.5003 | 0.5377 | 0.0327 | 2.8 | 5.9 | 35.6 |
| India | bm25:50 | 0.9134 | 0.8660 | 0.9108 | 0.1299 | 31.7 | 66.9 | 16.3 |
| India | bge_native:20 | 0.1680 | 0.1643 | 0.1680 | 0.1107 | 19.4 | 41.0 | 1387.5 |
| India | e5_native:20 | 0.1653 | 0.1614 | 0.1653 | 0.1085 | 19.5 | 41.1 | 171.2 |
| US | base | 0.9061 | 0.8502 | 0.9042 | n/a | n/a | n/a | 120.0 |
| US | base_wide:200 | 0.9427 | 0.8507 | 0.9042 | 0.0368 | 150.1 | 317.8 | 3.3 |
| US | addr_only:20 | 0.7110 | 0.6573 | 0.7110 | 0.0289 | 6.4 | 13.5 | 1.8 |
| US | name_noaddr:10 | 0.0378 | 0.0378 | 0.0378 | 0.0053 | 6.7 | 14.1 | 1.0 |
| US | conj:50 | 0.9861 | 0.9722 | 0.9857 | 0.0851 | 41.7 | 88.3 | 180.0 |
| US | rescue | 0.6227 | 0.5735 | 0.6227 | 0.0135 | 1.9 | 4.1 | 53.3 |
| US | bm25:50 | 0.9691 | 0.9392 | 0.9682 | 0.0672 | 33.8 | 71.6 | 23.5 |
| US | bge_native:20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | 15.4 |
| US | e5_native:20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 |

## Base misses: failure mode share and recovery rate by channel

**India**

| mode | share of base misses | base_wide | addr_only | name_noaddr | conj | rescue | bm25 | bge_native | e5_native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_script_name | 0.579 | 0.655 | 0.529 | 0.000 | 0.837 | 0.162 | 0.606 | 0.917 | 0.899 |
| common_name_crowd | 0.180 | 0.767 | 0.575 | 0.000 | 0.890 | 0.158 | 0.790 | 0.000 | 0.000 |
| empty_address_record | 0.061 | 0.302 | 0.000 | 0.512 | 0.423 | 0.036 | 0.383 | 0.000 | 0.002 |
| similar_but_outranked | 0.044 | 0.447 | 0.003 | 0.000 | 0.956 | 0.247 | 0.797 | 0.000 | 0.000 |
| alias_or_web_name_form | 0.042 | 0.543 | 0.207 | 0.000 | 0.592 | 0.247 | 0.559 | 0.000 | 0.000 |
| name_replaced_same_address | 0.030 | 0.713 | 0.528 | 0.000 | 0.894 | 0.194 | 0.713 | 0.000 | 0.000 |
| address_rewritten_name_kept | 0.024 | 0.520 | 0.034 | 0.000 | 0.743 | 0.063 | 0.446 | 0.000 | 0.000 |
| partial_name_and_address_noise | 0.018 | 0.346 | 0.000 | 0.000 | 0.617 | 0.075 | 0.609 | 0.000 | 0.000 |
| name_replaced_address_changed | 0.013 | 0.191 | 0.096 | 0.000 | 0.266 | 0.043 | 0.128 | 0.000 | 0.000 |
| no_key_survives_df_cap | 0.009 | 0.000 | 0.000 | 0.000 | 0.785 | 0.246 | 0.415 | 0.000 | 0.000 |

**US**

| mode | share of base misses | base_wide | addr_only | name_noaddr | conj | rescue | bm25 | bge_native | e5_native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| common_name_crowd | 0.422 | 0.452 | 0.557 | 0.000 | 0.996 | 0.122 | 0.859 | 0.000 | 0.000 |
| similar_but_outranked | 0.170 | 0.516 | 0.026 | 0.000 | 0.995 | 0.247 | 0.844 | 0.000 | 0.000 |
| empty_address_record | 0.162 | 0.251 | 0.000 | 0.347 | 0.534 | 0.077 | 0.356 | 0.000 | 0.000 |
| alias_or_web_name_form | 0.097 | 0.441 | 0.249 | 0.000 | 0.875 | 0.185 | 0.712 | 0.000 | 0.000 |
| name_replaced_same_address | 0.087 | 0.325 | 0.486 | 0.000 | 0.979 | 0.118 | 0.636 | 0.000 | 0.000 |
| no_key_survives_df_cap | 0.048 | 0.000 | 0.000 | 0.000 | 0.987 | 0.181 | 0.471 | 0.000 | 0.000 |
| name_replaced_address_changed | 0.008 | 0.115 | 0.192 | 0.000 | 0.846 | 0.038 | 0.308 | 0.000 | 0.000 |
| partial_name_and_address_noise | 0.006 | 0.222 | 0.000 | 0.000 | 1.000 | 0.056 | 0.389 | 0.000 | 0.000 |

## Multilingual retrieval details

```
{
 "India": {
  "bge_native": {
   "model": "BAAI/bge-m3",
   "dtype": "float16",
   "index": "exact GPU matmul (chunked)",
   "target": "pool records with Indic-script names",
   "revision": "5617a9f61b028005a4858fdac845db406aefb181",
   "dim": 1024,
   "pool_encoded": 752869,
   "pool_encode_sec": 1297.3,
   "queries_encoded": 10000,
   "query_sec": 18.8,
   "gpu_peak_gib": 3.11
  },
  "e5_native": {
   "model": "intfloat/multilingual-e5-small",
   "dtype": "float16",
   "index": "exact GPU matmul (chunked)",
   "target": "pool records with Indic-script names",
   "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
   "dim": 384,
   "pool_encoded": 752869,
   "pool_encode_sec": 157.6,
   "queries_encoded": 10000,
   "query_sec": 3.3,
   "gpu_peak_gib": 3.64
  }
 }
}
```