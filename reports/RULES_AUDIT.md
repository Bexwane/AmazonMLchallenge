# Rules Audit: Amazon ML Challenge 2026 (Business Entity Resolution)

Source of truth: `student_resource/README.md` and `utils/validate_submission.py` (read 2026-09-25).

| Item | Rule | Status / how we comply |
|---|---|---|
| Task | For every test Source-1 entity, list all matching Source-2/Source-3 records (0..many) | Pipeline outputs one row per test S1 |
| Target | `matched_entity_ids`, comma-separated S2-/S3- IDs | |
| Metric | F_0.5 per S1 entity, **macro-averaged**; singleton + empty prediction = 1.0; singleton + any prediction = 0.0 | `src/ber/metric.py`, unit-tested incl. official example (0.714) |
| Scored file | `matching_results.tsv` (TSV, header `source1_entity_id\tmatched_entity_ids`) | `io.write_id_lists` |
| Also required | `candidate_pairs.tsv`: the **final** candidate set the model scores; matches must be a subset | Written from the exact pairs the model scores |
| Format rules | every test S1 exactly once; no duplicate rows; no duplicate IDs in a list; only S2-/S3- IDs; IDs must exist in test | Enforced by the writer; checked with the official validator (`--check-ids`) |
| Country | Open set. Test adds **France** (not in train). Do not hard-code, filter or one-hot {US, India} | Blocking loops over whatever labels appear; `country` is **not** a model feature; abbreviation tables are merged across locales |
| Model licence | Final model must be MIT/Apache-2.0 and ≤ 8B parameters | v1 uses LightGBM (MIT) + rapidfuzz (MIT), no pretrained weights |
| External data | **Prohibited**: external databases, APIs, ER services, business registries, geocoding, internet augmentation | None used. Hand-written tables (street abbreviations, US state codes) are generic linguistic knowledge, not entity lookup. Listed in the methodology doc |
| Test data use | Not explicitly addressed | UNKNOWN → we only use test **inputs** unsupervised (IDF statistics for blocking within the test split). No test labels exist. Document it; revisit if organizers clarify |
| Final package | zip: `output/` (both TSVs), `code/business_entity_resolution/{src,README.md,requirements.txt}`, filled `Documentation_template.md` | To be assembled at freeze |
| Leaderboard | Public = subset of test; private = remainder; final ranking = private | Select by CV + stress CV, not public LB |
| Submission limits / deadline | UNKNOWN (not in README) | Team to confirm from portal |
| Compute limits | None stated | |
