# Architecture

PRISM is a Platform challenge service. It runs as a FastAPI application with SQLite state, internal Platform authentication, and GPU evaluation through the Platform Docker broker.

## High-Level Design

```mermaid
flowchart LR
    Miner[Miner] --> Proxy[Platform Proxy]
    Proxy --> Bridge[PRISM Bridge]
    Bridge --> DB[(SQLite)]
    Bridge --> Queue[Worker Queue]
    Queue --> Review[Static Review]
    Review --> LLM[LLM Review]
    LLM --> Broker[Docker Broker]
    Broker --> Eval[GPU Evaluator]
    Eval --> Scores[Scores]
    Scores --> Weights[get_weights]
```

## Main Components

| Component | Responsibility |
| --- | --- |
| FastAPI app | Public and internal HTTP routes |
| Repository | SQLite persistence for submissions, scores, sources, ownership, assignments |
| Worker | Claims pending submissions, reviews code, dispatches evaluation, finalizes scores |
| Component parser | Reads `prism.yaml`, separates architecture and training files, computes fingerprints |
| Container evaluator | Writes the project into a temporary workspace and runs it in an isolated container |
| Weights module | Converts architecture/training ownership into Platform-compatible hotkey weights |

## Platform Integration

Platform is responsible for miner-facing upload security. It verifies signatures, timestamps, nonces, and hotkey identity before forwarding a submission to PRISM.

PRISM receives verified submissions on:

```text
POST /internal/v1/bridge/submissions
```

The bridge trusts only internal Platform authentication and the verified hotkey header. Miner-supplied identity headers are not trusted.

## Execution Model

PRISM does not execute miner submissions directly in the master process. The worker performs static inspection and optional LLM review, then sends the project to an isolated evaluator container.

The current runtime path is GPU/broker oriented:

```text
PRISM worker -> DockerExecutor -> Platform Docker broker -> GPU evaluator container
```

Legacy local CPU and Lium-style execution are intentionally not part of the supported backend set.

## State Model

PRISM stores state in SQLite. Important tables include:

- `miners`
- `submissions`
- `eval_jobs`
- `scores`
- `submission_sources`
- `llm_reviews`
- `plagiarism_reviews`
- `architecture_families`
- `training_variants`
- `component_scores`
- `evaluation_assignments`

## Master and Validator Modes

The master can evaluate submissions via the broker-backed worker. PRISM also exposes internal validator-assignment routes so independent validators can be assigned reviewed submissions and return metrics.

```mermaid
sequenceDiagram
    participant V as Validator
    participant R as PRISM
    participant DB as SQLite
    V->>R: request assignment
    R->>DB: claim pending submission
    R-->>V: code and metadata
    V->>R: submit metrics
    R->>DB: finalize scores
```

## Failure Handling

Submissions can end in one of these states:

- `pending`
- `running`
- `completed`
- `failed`
- `rejected`

Rejected submissions fail review or contract validation. Failed submissions passed initial checks but failed evaluation or infrastructure execution.
