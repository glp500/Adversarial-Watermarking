# Real source documents

Openly licensed places to get genuine articles for `data/raw/`. Licences were checked on
2026-11-06. Re-check the licence on each article before use, because some publishers let
authors choose.

| Domain | Source | Licence | Good for |
|---|---|---|---|
| Biomedical and neuroscience | [PubMed Central Open Access subset](https://catalog.data.gov/dataset/pubmed-central-open-access-subset-pmc-oa) | Mixed CC. Filter the file list's License column to CC BY or CC0 | Bulk PDFs and clean XML. Use NCBI's FTP/OAI services only, since other bulk scraping is prohibited |
| Neuroscience | [eNeuro](https://www.eneuro.org/) (Society for Neuroscience) | CC BY 4.0 | Peer-reviewed neuroscience papers |
| Neuroscience, biology | [PLOS journals](https://journals.plos.org/plosone/s/licenses-and-copyright) (PLOS ONE, PLOS Biology, PLOS Comp Biol) | CC BY 4.0 for every article | Any domain, since PLOS ONE also publishes environmental and social-science work |
| Environmental | [Environmental Research Letters](https://doaj.org/toc/1748-9326) (IOP) | CC BY | Short climate and environment papers, which fit a 2B model's context |
| Environmental | [Hydrology and Earth System Sciences](https://hydrology-and-earth-system-sciences.net/policies/publication_policy.html) and other Copernicus/EGU journals | CC BY 4.0 | Earth-system science |
| Political science | [Politics and Governance](https://www.cogitatiopress.com/politicsandgovernance) (Cogitatio) | CC BY 4.0 | Peer-reviewed political science |
| Political science | [Research & Politics](https://journals.sagepub.com/home/rap) (SAGE) | Author's choice, often CC BY-NC | Short articles. Check each one |
| News | [Wikinews](https://en.wikinews.org/) and the [Wikinews-multilingual](https://www.huggingface.co/datasets/Fumika/Wikinews-multilingual/commit/bef97ff4c06133ad90e221e75046046e3cf68c7b) dataset | CC BY 2.5 | Real news stories, already in text form |
| News | [Global Voices](https://globalvoices.org/about/global-voices-attribution-policy/) | CC BY 3.0, adaptation allowed | International news and citizen reporting |
| News commentary | [The Conversation](https://theconversation.com/republishing-guidelines) | CC BY-ND 4.0 | Academics writing news-style. **No derivatives**, so rewritten versions must stay internal and unpublished |

## Practical notes

- **Length.** The rewrite prompt sends the whole article to the generator. Full papers run to
  8,000 words or more, so use the abstract plus introduction, or a news-length section.
- **Getting text in.** `wm-eval ingest` converts files into the JSONL the runner reads:

  ```bash
  wm-eval ingest downloads/environment/ --out data/raw/environment.jsonl \
      --domain environment --max-words 2000 --licence CC-BY-4.0
  ```

  It accepts `.pdf`, `.html` (a saved web page), `.txt` and `.md`. It removes invisible
  characters, drops a trailing reference list, re-joins hyphenated words and records the
  file, URL, licence and word counts in each document's `provenance`.
- **Prefer the web page to the PDF.** In a test on the PLOS ONE "whale pump" paper, the PDF
  turned "2.3 × 10⁴" into "2.3 6104" and mixed author affiliations into the text. The saved
  HTML page kept the numbers. Site boilerplate can still slip through, so read each
  document in the JSONL before running. Scanned PDFs have no text layer and need OCR first.
- **Licences and this project.** Rewriting an article creates a derivative work. Keep sources
  CC BY or CC0 where possible, record the licence and URL for each document, and never publish
  generated rewrites as news.
- **Domain field.** Set `domain` to `news`, `environment`, `politics` or `neuroscience`. Then
  `tables/comparisons_by_domain.csv` will split results by field.
