# Evaluation suite (for the paper)

```bash
python -m eval.run_eval                     # E1–E4 on seeded synthetic data (~1 min) -> eval/results/
GEMINI_API_KEY=... python -m eval.run_eval  # also runs the LLM baseline for E1
python -m eval.real_eval template eval/real_decks   # then hand-label eval/labels.csv
python -m eval.real_eval score eval/real_decks      # real-deck classification table
```
Outputs: `results/results.md` (readable), `results/results.json` (raw), `results/tables.tex` (paste into LaTeX).

## Experiments
| ID | Question | Metric | Data |
|---|---|---|---|
| E1 | Does the rule-based checker catch planted numerical contradictions without false alarms? | P/R/F1 per type, false-alarm rate, ms/deck | 1,500 synthetic decks, 5 contradiction types + clean |
| E1b | Is it better / cheaper than asking an LLM? | deck-level F1, s/deck | same generator, Gemini Flash |
| E2 | Does the position signal help slide classification? | accuracy with 95% bootstrap CI, macro-F1 | synthetic sanity check; **real hand-labeled decks** via `real_eval.py` |
| E3 | Can Deck DNA recover an industry's slide order, and how many decks does it need? | Kendall τ against the true order vs. #decks | synthetic industry "house styles" plus random swaps |
| E4 | Does the ingest pipeline scale, and is it idempotent and privacy-preserving? | pages/s, speedup, re-ingest time, PII leaked | 10–100 generated PDFs (15 pages each) |

## Findings so far (8-CPU laptop)
- **E1:** macro-F1 1.000, 0% false alarms on clean decks, 0.03 ms per deck. *The first version had 51% false alarms*: the pre-revenue check flagged big year-3 targets. Restricting it to year-1 revenue fixed it. Report this before/after in the paper, since it's an honest error analysis.
- **E3:** Deck DNA reaches τ≈0.97 with only 3 decks per industry and 1.0 by 20. The fixed canonical template scores 0.73.
- **E4:** the first version was *slower* in parallel (0.04× the single-worker speed) because it started a fresh process pool on every call. A persistent pool plus running small batches (<8 docs) in one process gives a 4.8–5.4× speedup and about 5,400 pages/s. Re-ingesting takes about 1 ms per 100 decks. 0 emails leaked.
- **E2 (synthetic):** every method scores 98–100%, so the synthetic data is too easy to separate them. **Use the real-deck results in the paper**, not these.

## Threats to validity (state these in the paper, because reviewers will raise them)
1. **E1 is circular on synthetic data:** the same authors wrote both the generator and the checker, so 100% shows *correctness*, not *generality*. Mitigations: (a) the LLM baseline on the same data; (b) hand-annotate contradictions in 30–50 real or student decks and report on those.
2. **E2 synthetic results are saturated**, so you need real, hand-labeled decks. Aim for 25+ decks (about 300+ pages), with a second annotator on a subset to report Cohen's κ.
3. E3 uses simulated orders. Validate on the real corpus by comparing the learned order with published advice (Sequoia, YC, Guy Kawasaki).
4. Throughput depends on hardware and the PDF generator. Report the CPU, and note that scanned PDFs (which need OCR) aren't covered.

## Suggested paper outline
**Title:** *Grounded and Verified: Retrieval-Augmented Pitch Deck Generation with Cross-Slide Numerical Consistency Checking*
1. Introduction: LLM pitch tools produce fluent but unverified numbers.
2. Related work: retrieval-augmented generation (RAG), hallucination detection, numerical reasoning in documents, slide/document layout classification.
3. System: ingest pipeline → slide classifier → retrieval → drafter/critic → verifiers (figure: the diagram in the top-level README).
4. Method: consistency rules (formalize each as a constraint, e.g. |ask/burn − runway| / runway > 0.35), the position-weighted classifier, Deck DNA (mean normalized position).
5. Experiments: E1–E4 plus the real-deck E2.
6. Error analysis: the pre-revenue false-alarm fix and the process-pool fix.
7. Limitations and future work: OCR, learned (rather than hand-written) consistency constraints, a user study with founders and investors.
