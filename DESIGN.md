# Design Write-up: QTrade AI Support Assistant

## What I built and why

The challenge asked for a support assistant grounded in QTrade's help documentation — retrieve relevant content, answer with citations, escalate when it should not be answering. The interesting decisions were in the tradeoffs between what works for a four-doc demo and what is correct for a production marketplace system. I used the latter as the governing constraint throughout.

---

## Pipeline
Every customer message flows through the same steps, in order:

```
Customer query
      |
      v
[1] Escalation pre-check     <-- regex: human request, repeat frustration, safety hazard
      | (no trigger)
      v
[2] ChromaDB retrieval       <-- bge-small-en-v1.5, cosine similarity, top-k chunks
      |
      v
[3] LLM generation           <-- strict grounding prompt, conversational noise handled here
      |
      v
[4] LLM output check         <-- if [Escalate] in response, escalate
      |
      v
AssistantResponse            <-- answer + cited_docs (parsed from LLM output) + escalation state
```

Each step has one responsibility. The escalation logic is completely separate from the generation logic, the retrieval layer knows nothing about the LLM, and the assistant orchestrator is the only place that knows how everything connects. This means any individual component can be swapped without touching the others — which is how I ended up supporting both Gemini and Ollama through the same pipeline without duplicating any logic.

### How the files connect

`document_loader.py` reads the `.txt` help docs and produces `DocumentChunk` objects — the atomic unit that everything else operates on. Each chunk carries its source document name, file stem, and character offset so citations are always traceable. The source document name is taken from the `Doc:` header when available, otherwise the file stem is used. This ensures that documents can be added to the docs directory without requiring special formatting.

`vector_store.py` takes those chunks, embeds them with `bge-small-en-v1.5`, and upserts them into a ChromaDB collection. On retrieval it embeds the query with the BGE retrieval prefix and returns ranked `RetrievedChunk` objects with cosine similarity scores. The collection is configured for cosine distance at creation time.

`rules.py` evaluates a message against four explicit regex patterns and returns an `EscalationDecision`. It handles human requests, repeat frustration, and the no-grounded-answer signal.

`answer_generator.py` takes retrieved chunks and a query, builds a grounding prompt, calls whichever LLM provider is configured, and parses the response. The cited docs in the returned `GeneratedAnswer` are extracted by scanning the LLM's output for `[Source: ...]` tags — not from the full retrieved set — this was a bug I caught during testing where Warranty was showing up as a source for SmartHub reset questions because it was in the retrieved pool even though the LLM never used it.

`assistant.py` is the orchestrator. It wires everything together, runs the five steps in sequence, and returns a consistent `AssistantResponse` regardless of which path was taken.

`cli.py` is the entry point. Interactive REPL, single-query mode, and a `--run-samples` flag that runs all six Appendix B queries at once.

`eval_harness.py` runs 8 labelled queries through the full pipeline and scores them across three dimensions. It saves timestamped JSON results so runs are comparable across model changes.

---

## Key technical decisions and why I made them

### Embedding model: bge-small-en-v1.5 over MiniLM

I initially considered using `all-MiniLM-L6-v2` because it is the default safe choice with 22 MB, MIT licence, runs on CPU. But QTrade's support corpus is keyword-dense: model names, percentages, day counts, error codes. MiniLM was trained for general semantic similarity on Wikipedia and NLI datasets, not retrieval tasks. It compresses specific technical terms into the same neighbourhood as semantically similar but factually different concepts.

Whereas, `bge-small-en-v1.5` is trained specifically for retrieval via contrastive learning and consistently outperforms MiniLM by 8-12 points on MTEB retrieval benchmarks. It also benefits from a query-side prefix (`Represent this sentence for searching relevant passages:`) that was part of its training — skipping this measurably hurts retrieval accuracy and is easy to miss. I already knew this model from my [Scientific Paper QA project](https://github.com/aeesh/paper-qa-system) and had validated it there on the same retrieval task. At production scale with hundreds of docs the gap between MiniLM and BGE compounds. The upgrade path from here is `bge-large-en-v1.5` (1.3 GB, same interface) or a domain-fine-tuned model evaluated against actual QTrade query logs.

### Vector store: ChromaDB over in-memory numpy

For this I first considered using a numpy matrix with L2-normalised dot products for cosine similarity. While it will work at the scale of this submission, it'd be the wrong architecture for production: every process start re-embeds everything from scratch, and there is no incremental update path — adding a new help doc requires rebuilding the entire index. ChromaDB's `PersistentClient` solves both problems. It upserts incrementally, survives restarts, and uses HNSW under the hood for approximate nearest-neighbour search that scales without degradation.

### Why cosine and not L2

The distance metric is set to cosine at collection creation time instead of L2 because it measures the similarity of vector direction rather than absolute distance. In text retrieval, two chunks can have the same meaning but different lengths, and cosine still considers them similar, whereas L2 can penalise the longer chunk due to differences in vector magnitude.

### Retrieval top-k: 3

The choice of k is a genuine tradeoff that cannot be resolved without knowing the corpus size.

**Too low (k=1):** one miss cases where the answer spans two related chunks — a returns policy that states the window in one sentence and the fee in the next could be split across chunks, and k=1 would retrieve only one half.

**Too high (k=7+):** injects noise. Chunks that scored high enough to be retrieved but are not actually about the query end up in the context window. The LLM then has to reason across irrelevant content, which increases the chance of a confused or hallucinated answer.

**The corpus size problem with k=5:**  With four docs in the current corpus, k=5 fills the context window with chunks from unrelated docs just because they had nowhere else to rank. With a larger corpus, k=5 is appropriate because positions 3-5 would be filled with genuinely related content. I settled on k=3 for the submission because it keeps the context clean at the current doc count, the answers are fully grounded in the top 1-2 chunks anyway, and the cited docs display is cleaner. The `--top-k` CLI flag makes this easy to change without touching code.

### Chunking strategy: sentence-aware sliding window

The chunking is a hybrid between fixed-size and pure sentence chunking. It tries to break on sentence boundaries before hitting a character limit, with overlap to prevent cold cuts. This is not the same as token-based chunking as it counts characters and not tokens. Token-based chunking (what LlamaIndex's `SentenceSplitter` does with `chunk_size=512`) is strictly more correct for RAG systems because LLMs process token windows, not character windows. A 512-token chunk and a 512-character chunk are very different things, and the mismatch between how the chunker counts and how the LLM processes can cause truncation or inefficiency.

The reason I didn't switch to token-based chunking here is that the current docs are small enough that the distinction doesn't affect output quality, and implementing token-based chunking without a framework dependency requires pulling in a tokenizer. The chunking strategy is calibrated for correctness at this doc size.

At production scale, an upgrade to a token-based strategy with a window of around 512 tokens and overlap of 50 tokens would be better. Also, hierarchical chunking (splitting on document structure like headers and sections) or semantic chunking (grouping sentences by embedding similarity) would further improve retrieval on structured support documentation.


### Escalation: rule-based

Four triggers are implemented in `rules.py`, all using regex:

- `SAFETY_HAZARD` — physical danger language. QTrade sells hardware, so a customer reporting fire, smoke, or sparks from a device is a liability event and potential recall signal. Needs immediate human routing.
- `EXPLICIT_HUMAN_REQ` — customer explicitly asked for a person.
- `REPEAT_FRUSTRATION` — language indicating repeated contact or high distress.
- `NO_GROUNDED_ANSWER` — set when the LLM cannot answer from the retrieved context and outputs `[Escalate]`, meaning the question is outside the documented topics.

Regex was chosen because these triggers involve explicit, unambiguous phrasing that any engineer can read the patterns in `rules.py` and know exactly what fires and why. This is important for support operations where escalation policies are often maintained by non-engineers and need to be auditable.

The known limitation is the safety trigger. The pattern space for physical danger is unbounded — "fire", "my device gets really warm", "there's a weird smell" are all valid hazard signals that no finite regex set covers completely. Three alternatives were evaluated:

1. **Expanded regex** — add patterns for heat, smell, temperature, fire vocabulary. Zero extra cost, deterministic, but fundamentally bounded.
2. **Separate LLM classifier call** — binary HAZARD/SAFE prompt before the main generation step. Generalises across all phrasings naturally, but adds a full LLM round-trip to every message regardless of whether it is dangerous, which is a meaningful latency cost.
3. **Piggyback on the main generation call** — instruct the LLM to return a safety verdict alongside the answer in one structured response. No extra cost, but safety detection happens after retrieval instead of before it.

The current implementation uses expanded regex (option 1) to avoid extra API calls. Option 3 is the most cost-efficient production improvement with one structured call returning both answer and safety classification.

### Conversational noise: handled in the prompt

Greetings, vague statements, and expressions of frustration produce low retrieval scores across all docs. Without prompt intervention, the LLM would correctly identify these as unanswerable from context and output `[Escalate]`. Rule 2 in the system prompt instructs the LLM to respond warmly to conversational messages and invite the customer to share their issue.

This was handled in the prompt rather than in code because the LLM's semantic understanding is better suited to the distinction than any code-level filter. "I am not certain" is conversational noise on its own but a genuine question in the right context where regex cannot make that call.

One failure mode remains: the assistant is stateless. "I am not certain" as a follow-up to "how do I reset my hub?" in a multi-turn conversation would be interpreted by a human as follow-up ambiguity. The assistant has no conversation history, so it treats it as a standalone vague statement. The correct fix is multi-turn session tracking, instead of more prompt rules.

### Cited docs: parsing LLM output vs. showing the retrieved set

The `cited_docs` field in `AssistantResponse` is populated by parsing `[Source: ...]` tags from the LLM's response text rather than from the full set of retrieved chunks.

The retrieved set approach was the initial implementation and produced misleading output. A SmartHub reset question at k=3 retrieves the SmartHub chunk as the top result and then fills positions 2-3 with Warranty and shipping chunks because with four docs those are the next closest chunks available. A user seeing "Sources: SmartHub Setup & Troubleshooting, Warranty, Shipping" for a reset question would reasonably wonder why warranty information was relevant, and might distrust the answer.

A score-threshold approach was also tested but failed in practice: the score distribution is not stable across queries, so a fixed cutoff that filters correctly for one query over-filters for another.

Parsing what the LLM explicitly cited is more honest as it shows only what the model drew on.

### Handoff summary and real-world escalation integration

Every escalated response includes a structured handoff summary with the customer's question, the trigger reason, retrieved context scores, and a recommended routing action. In the current implementation this prints to the CLI — nothing actually notifies a human agent, because the right integration depends on what support infrastructure QTrade runs. In production this would trigger a webhook to a ticketing system (Zendesk, Freshdesk), a Slack notification, or a live chat handoff. The summary format is machine-readable by design so it can be consumed by any of these without changing the internal representation.

---

## Evaluation

### Why these three metrics

Three dimensions per case:

**Escalation accuracy** — does the assistant escalate exactly when it should and not when it should not? This is the most important metric. A false negative (not escalating when you should) is a customer experience failure. A false positive (escalating when you should not) undermines the point of having an assistant at all. Getting this right is the core design challenge.

**Grounding** — does the non-escalated answer cite a real document? This is a proxy for whether the LLM used retrieved context rather than generating from memory. A grounded answer that is wrong is a retrieval failure. An ungrounded answer that is correct is a hallucination that happened to be right. Both are bad; grounding catches the structural failure.

**Keyword accuracy** — does the answer contain the expected key fact? For support responses this matters most for numerical precision: "15% restocking fee", "5-7 business days", "30 days". These are the exact values a customer needs and the ones most likely to be hallucinated incorrectly.

### Results

| Provider | Pass rate | Escalation accuracy | Grounding | Keyword accuracy |
|---|---|---|---|---|
| Gemini (models/gemini-3.1-flash-lite) | 8/8 (100%) | 100% | 100% | 100% |
| Ollama (llama3.2 3B) | 8/8 (100%) | 100% | 100% | 100% |

The identical scores across a commercial API and a 3B local model confirm that the hard work is in retrieval and escalation — by the time the LLM gets the context, the question is already narrow enough that even a small model answers it correctly.
### Limitations of the eval dataset

Eight cases is a minimum viable eval. The escalation cases pass at 100% because they test exactly the patterns the implementation was built to handle with no adversarial testing. A more rigorous eval would include paraphrased danger signals ("this device is producing heat and an unusual odour"), compound queries spanning multiple docs, and ambiguous partial-answer cases. The keyword scorer is a heuristic as it cannot catch semantic errors, "5-7 weeks" instead of "5-7 business days" passes because "5" and "7" are present. An LLM-as-judge layer would catch this at the cost of additional API calls per eval run.

---

## What a different escalation design would look like

An alternative is **confidence-based escalation using the LLM itself**: after generating an answer, prompt the LLM to self-assess confidence on a 1-5 scale and escalate below 3. This catches cases where retrieval succeeded topically but the chunks do not fully answer the question, and it handles novel phrasing that regex misses.

The tradeoffs are significant. Confidence self-assessment in LLMs is unreliable, smaller models are known to be overconfident. The approach also adds latency and API cost on every response, and it gives the LLM operational authority over escalation decisions in a place where determinism is preferable.


The production answer is a hybrid: deterministic rules for explicit triggers, the LLM's own `[Escalate]` output signal for the no-grounded-answer case (which the current implementation already does), and a structured generation format that piggybacks safety classification onto the main call for the safety trigger.

---

## Scale considerations

**At 10x docs (40 docs):** current stack holds fine. ChromaDB's HNSW index handles this without configuration changes. k should be revisited upward, positions 3-5 start containing genuinely related content at this corpus size.

**At 100x docs (400 docs) with concurrent users:** embedding runs synchronously on the main thread, under load this needs  batch processing or a worker pool. ChromaDB local client should move to a hosted cluster. Dense-only retrieval starts missing exact keyword matches, adding BM25 sparse retrieval with reciprocal rank fusion plus a cross-encoder reranker for the final top-k, is the standard production upgrade. Chunking should shift to token-based (512 tokens, 50 overlap), and hierarchical or semantic chunking to preserve document structure.

**Deployment and monitoring:** the assistant wraps into a FastAPI service behind an authenticated endpoint, containerised with Docker. Logging captures every query, retrieval scores, escalation trigger, and LLM latency. Alerting fires on escalation rate spikes (suggests a new query category the docs do not cover, or a product issue generating support volume) and on answer latency exceeding 3 seconds. Quality regression is caught by running the eval harness as a CI step on every doc update — a drop below 90% pass rate blocks the deployment.