# Data Autopsy (T0, 2026-09-25)

Scripts: `src/eda/{gt_stats,file_stats,show_groups,hard_negatives,distractor_addr,id_leakage}.py`.

## Sizes and schema
All files: `entity_id, business_name, business_address, country`, tab-separated, 0 malformed rows.

| file | rows | countries | empty addr | Indic-script names |
|---|---|---|---|---|
| train S1 | 2,206,821 | US 1.32M, India 0.88M | 0 | 0 |
| train S2 | 5,034,616 | US 3.02M, India 2.02M | 168,967 (3.4%) | 474,345 (23% of India) |
| train S3 | 5,285,603 | US 3.17M, India 2.12M | 175,916 (3.3%) | 278,524 (13% of India) |
| test S1 | 1,732,544 | US 0.66M, **France 0.26M**, India 0.81M | 0 | 0 |
| test S2 | 4,887,273 | India 2.31M, France 0.70M, US 1.87M | 129,408 | 546,606 |
| test S3 | 5,082,316 | India 2.41M, France 0.73M, US 1.95M | 136,098 | 320,639 |

S1 is clean (Title Case, full addresses, no empties). S2 is UPPER-CASE addresses with native-script
Indian names and state names; S3 has abbreviated states (WB, MH, UP), typos and reordering.

## Ground truth
* Matches per S1: 0 → 5.6% (123,247 singletons); mode 3; mean ≈ 3.46; max 11.
* S2 matches per S1: 0–5; S3: 0–6.
* **Every matched S2/S3 record belongs to exactly one S1** (0 duplicates across 7.6M matched IDs), so the
  ground truth is a partition. We exploit this with exclusive assignment.
* 26.6% of S2 (1.34M) and 25.4% of S3 (1.34M) records match **no** S1. These are distractors. Most
  belong to entities absent from S1 and have no near look-alike in S1. A minority are deliberate hard
  negatives: a different business at the same address ("Onyxnex Quality Usa" vs "Baix Quality Usa", 184
  Armstrong Dr, Spindale).
* 0 cross-country matched pairs (1% sample), so blocking within a country is safe.
* **Train/test shift:** S2+S3 records per S1 = 4.68 (train) vs **5.76 (test)**. Either more matches or
  more distractors per S1 in test. This is a precision risk; watch the LB-CV delta.

## Noise patterns seen in matched groups
Names:
* legal-suffix swaps and moves ("Llc Velex", "Pvt Usa Vision Ltd")
* appended "Center"/"Service(s)"/"Inc"
* `(ID: 96415)` and `#70318` tags
* website/handle forms ("literacycenter.com", "@willoughbycertified", "#oncologygulf")
* aliases: "Rizacira formerly: X", "Veranex f/k/a X", "Pyrairibrix t/a X"
* leet typos (0/o, 1/l, 5/s)
* letter typos, word-order swaps
* native Devanagari/Bengali/Kannada spellings of the English name
* occasionally a completely different random name at the same address that is still a TRUE match
  ("Irinyla" ↔ Velex LLC)

Addresses:
* Street/St/**Saint**, Road/Rd, Eleventh/11th, leading zeros (0520), "##801"
* `NULL` / `<NULL>`, "PMB 9914", appended "CITY", component reordering
* full vs abbreviated states, missing components
* India C/O and Near-landmark forms

France (test only):
* R./R/Av/Q./All./PL abbreviations
* region vs département (Nouvelle-Aquitaine vs Gironde)
* bis/ter, Nº/N°, accents
* bracketed legal forms like [S.A.R.L.] and (S.A.S.)
* few distinct cities, so a city key alone is weak

## Implications
1. Blocking must use **name and address** keys. Address rescues random-name and native-script cases;
   the name separates same-address hard negatives.
2. Indic romanization + a phonetic consonant skeleton is required for native-script names.
3. Precision comes from competition: an S2/S3 record that matches another S1 better is not ours
   (exclusivity + reverse-rank features).
4. France is unseen. Keep features country-agnostic and use a country-holdout stress CV as the proxy.
