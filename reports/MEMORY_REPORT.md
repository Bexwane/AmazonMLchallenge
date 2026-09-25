# Memory Report (Kaggle limit 30 GiB)

| run | peak RAM | runtime | note |
|---|---|---|---|
| E003 all-S1 blocking eval | 21.2 GiB (children 0.21) | 17.6 min | blocking 11.7 min |
| E007 all-S1 blocking eval | 27.07 GiB (children 0.2) | 106.7 min | blocking 95.9 min |
| probe India (one process) | 15.97 GiB (children 0.1) | 31.2 min | sampled queries, all channels |
| probe US (one process) | 18.32 GiB (children 0.1) | 6.9 min | sampled queries, all channels |

## Per channel (probe, RSS after the channel ran, cumulative within the process)

| country | channel | RSS after (GiB) | peak so far |
|---|---|---:|---|
| India | base | 8.81 | 14.02 GiB (children 0.1) |
| India | base_wide:200 | 8.91 | 14.02 GiB (children 0.1) |
| India | addr_only:20 | 9.09 | 14.02 GiB (children 0.1) |
| India | name_noaddr:10 | 9.09 | 14.02 GiB (children 0.1) |
| India | conj:50 | 9.98 | 14.02 GiB (children 0.1) |
| India | rescue | 11.08 | 14.02 GiB (children 0.1) |
| India | bm25:50 | 11.53 | 14.02 GiB (children 0.1) |
| India | bge_native:20 | 11.17 | 15.97 GiB (children 0.1) |
| India | e5_native:20 | 11.76 | 15.97 GiB (children 0.1) |
| US | base | 12.81 | 14.8 GiB (children 0.1) |
| US | base_wide:200 | 13.02 | 14.8 GiB (children 0.1) |
| US | addr_only:20 | 13.18 | 14.8 GiB (children 0.1) |
| US | name_noaddr:10 | 13.18 | 14.8 GiB (children 0.1) |
| US | conj:50 | 14.2 | 16.98 GiB (children 0.1) |
| US | rescue | 15.93 | 17.27 GiB (children 0.1) |
| US | bm25:50 | 16.38 | 18.32 GiB (children 0.1) |
| US | bge_native:20 | 16.95 | 18.32 GiB (children 0.1) |
| US | e5_native:20 | 16.95 | 18.32 GiB (children 0.1) |

Rules kept: no N x M similarity matrix (sparse top-k per query chunk); dense embeddings only for the native-script subset in fp16 on the GPU; one process per country; parquet caches.