# Security Model

PRISM evaluates untrusted miner code. Its security model assumes submissions may be malicious and separates identity verification, static review, LLM policy review, duplicate review, and execution isolation.

## Identity and Upload Security

Miner-facing uploads are handled by Platform. Platform verifies:

* hotkey identity
* signatures
* timestamps
* nonces
* request freshness
* challenge routing

After verification, Platform forwards the payload to PRISM through:

```text
POST /internal/v1/bridge/submissions
```

PRISM trusts the verified hotkey header only on authenticated internal requests.

## Internal Authentication

Internal endpoints require the shared Platform challenge token:

```text
Authorization: Bearer <shared-token>
```

The token is read from `PRISM_SHARED_TOKEN`, `CHALLENGE_SHARED_TOKEN`, or a configured secret file. Production configs should use secret files instead of inline values.

## Static Review

Before evaluation, PRISM inspects Python code for:

* forbidden imports
* forbidden calls
* unsafe attributes
* invalid top-level code
* missing contract functions
* unsupported project structure

The review is applied to all Python files in the submitted project. The entrypoint must satisfy the model contract, while helper files may be contract-free but still pass safety checks.

## Evidence-Gated LLM Policy Review

PRISM's LLM review flow is evidence-gated. The reviewer must call `submit_mermaid` before `submit_verdict`. A verdict before Mermaid evidence is rejected as an ordering error.

LLM rejection requires deterministic evidence. When the LLM is suspicious but does not provide deterministic evidence, PRISM records quarantine or hold audit state instead of treating suspicion as a final rejection. This policy is controlled by `llm_review_policy.evidence_required_for_rejection`, which defaults to `true`.

The evidence requirement is motivated by benchmark-contamination and model-security literature. **Rethinking Benchmark and Contamination for Language Models with Rephrased Samples** (Yang et al., 2023) shows that public benchmark overlap can survive simple string matching and can make suspiciously high scores ambiguous. **BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain** (Gu, Dolan-Gavitt, and Garg, 2017/2019) shows that models can preserve malicious trigger behavior while passing normal validation. PRISM therefore requires concrete, reproducible evidence before a review can reject a miner submission for malicious or dishonest behavior.

The default review intent is to catch code that attempts:

* secret exfiltration
* filesystem or process escape
* network escape
* hidden behavior unrelated to the model contract
* plagiarism or copied miner code

LLM review can be advisory or required depending on operator configuration.

## Plagiarism, Similarity, and Quarantine

PRISM stores source snapshots and can compare submissions against prior code and canonical architecture graphs. Deterministic duplicate reports may produce these outcomes:

| Outcome | Meaning |
| --- | --- |
| `reject` | Exact source duplicate or deterministic policy violation. |
| `attach` | Same architecture graph with source changes that attach as an implementation variant. |
| `quarantine` | Suspicious similarity or ambiguous graph match needing review. |
| `allow` | No blocking similarity found. |

Quarantine is suspicion-only protection. It prevents a questionable submission from affecting ownership or weights until review resolves it.

## ZIP Hardening

ZIP extraction rejects:

* symlinks
* path traversal
* unsafe paths
* unsupported file types
* excessive file count
* excessive total bytes

This prevents archive-level attacks before code review begins.

## Execution Isolation

PRISM does not execute submitted code inside the API or worker process. Evaluation happens in containers through the Platform Docker broker for GPU paths. Local CPU smoke runs are non-scoring wiring checks.

Container limits include:

* CPU quota
* memory and swap limits
* PID limits
* network policy
* read-only runtime option
* optional GPU type and count

Official score-eligible GPU jobs use the fixed official GPU profile from SQL runtime config or defaults, with a maximum of 8 GPUs. Autosplit is for non-scoring or development paths only.

## Scientific Security References

| Area | Study | PRISM implication |
| --- | --- | --- |
| Supply-chain attacks | Gu, Dolan-Gavitt, and Garg, 2017/2019, *BadNets* | Treat submitted code and artifacts as adversarial even when validation metrics look normal. |
| Benchmark contamination | Yang et al., 2023, *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples* | Do not treat anomalous public benchmark scores as sufficient evidence without contamination analysis. |
| Model documentation | Mitchell et al., 2019, *Model Cards for Model Reporting* | Require structured model/run metadata for evaluator artifacts. |
| Dataset documentation | Gebru et al., 2018/2021, *Datasheets for Datasets* | Track dataset provenance, transformations, and intended use. |
| Long-context evaluation limits | Liu et al., 2023, *Lost in the Middle*; Hsieh et al., 2024, *RULER* | Use long-context benchmarks as controlled evidence, not as a single safety verdict. |

## Operational Guidance

* Use real secret files in production, not inline tokens.
* Keep public submissions disabled when PRISM is deployed only behind Platform.
* Run broker-backed GPU evaluation for official GPU paths.
* Enable LLM review for production subnet operation when policy requires it.
* Monitor rejected, quarantined, held, and failed submissions separately.
