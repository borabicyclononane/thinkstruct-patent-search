# Independent-claim labels in the large-dataset sample: checked against Google Patents

The large-dataset sample (`experiment-big-data/sample.py`) is not scraped. A claim counts as independent if
its text cites no other claim. This is the scraper's text rule (`TEXT_REF` / `BROKEN_REF` in
`scraper/google_patents.py`), applied to the dataset's own claim text. To check it, 200 random patents from
`sample_10000` were scraped from Google Patents (`check_independence.py --n 10000 --check 200`, seed 0, 0 scrape
errors). Each claim was then compared with three references:

| reference | what it is | claims compared | agreement |
|---|---|---|---|
| scraper | `<claim-ref>` tags OR the text rule on Google's text (ground truth for the small dataset) | 3,394 | 100% |
| claim-ref | `<claim-ref>` tags only (structural, no text rule) | 3,394 | 100% |
| google | Google's own `claim-dependent` marking | 3,394 | 100% |

There were 0 claims the rule wrongly put in the index and 0 it wrongly left out. 453 of the 3,394 claims are
independent (2.3 per patent).

Edge cases, which explain the 47 rows in `independence_disagreements.csv`:
- `20050015886` numbers its claims "45)" instead of "45.". `sample.py`'s number pattern misses this, so its
  claims were numbered 1-23 by position instead of 45-67. Compared by position, all 23 labels agree. The labels
  are right, but the stored claim number is wrong and the text keeps its "45)" prefix. This affects 26 of 24,466
  independent claims (14 patents) in `sample_10000`, and 514 of 242,989 (212 patents) in `sample_100000`.
  It is not fixed.
- `20240338341`: Google lists an empty claim 6. The dataset's claim 6 is empty too, so `sample.py` drops it.

## End to end: are all independent claims in the index?

The check above compares labels by claim number. To also check the text the index actually embeds
(`sample_10000.jsonl`, after cleaning and dropping stubs), each patent's sample claims were paired one-to-one
with Google's independent claims by full-text similarity, so a numbering problem can't hide a miss
(`independence_end_to_end.csv`):

| | |
|---|---|
| Google independent claims | 456 |
| found in the sample | **456 (recall 100%)** |
| sample claims that aren't independent on Google | 1: an exact duplicate (the dataset lists claim 1 of `20230052527` twice; 6 of the 10k patents have a duplicated claim) |
| text similarity of matched pairs | median 1.0, 95% identical, minimum 0.90 |

The text differs only in reference-numeral spacing ("( 16 )" vs "(16)"), HTML entities, broken encoding on
Google's side, and in two claims some wording from a slightly different version of the claims. The openings,
claim references and preambles are all intact. So the large dataset doesn't have the small dataset's damage
(lost numbers, preambles and "of claim N" text).

Conclusion: on clean claim text, the text rule reproduces the structural claim references exactly, so the
sample's independent claims can be used without scraping.
