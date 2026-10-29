# InspectAI architecture and technical contracts

## Target component flow

```mermaid
sequenceDiagram
 participant Q as Quality operator
 participant UI as Next.js UI (planned)
 participant API as FastAPI
 participant W as Inspection worker (planned)
 participant O as Object store (planned)
 participant DB as PostgreSQL (planned)
 Q->>UI: Select board design; upload image
 UI->>API: POST /inspections
 API->>O: Store original image
 API->>DB: Create queued job
 API-->>UI: job id
 API->>W: Enqueue job
 W->>O: Fetch reference + candidate
 W->>W: Validate; align; classical/deep inference
 W->>O: Save mask and overlay
 W->>DB: Save score, boxes, model version, trace
 UI->>API: GET /inspections/{id}
 API-->>UI: Results + review status
 Q->>UI: Approve / reject prediction
 UI->>API: POST /inspections/{id}/feedback
 API->>DB: Store audited feedback
```

**Today:** only synchronous `POST /inspect` with image files exists. The sequence above is a design target, not a description of current behavior.

## Proposed entities

- `board_designs`: `id`, `name`, `revision`, `reference_image_uri`, `created_at`.
- `inspection_jobs`: `id`, `board_design_id`, `candidate_uri`, `status`, `submitted_at`, `completed_at`, `model_version`, `score`, `latency_ms`, `warnings`.
- `detections`: `id`, `job_id`, `bbox_xywh`, `mask_uri`, `anomaly_score`, `class_label_nullable`.
- `reviews`: `id`, `job_id`, `reviewer_id`, `decision`, `corrections_json`, `reviewed_at`.
- `model_versions`: `id`, `algorithm`, `data_version`, `config`, `checkpoint_uri`, `evaluated_at`.

## Deployment phases

1. **Local:** Python/OpenCV, FastAPI, no persistent DB. CPU.
2. **MVP:** Docker Compose, Postgres, Redis, worker, object store emulator, Next.js.
3. **Pilot-ready research:** authenticated users, quotas, access controls, backups, audit logs, observability, model registry, reproducible offline evaluation, documented failure modes.

## Security and reliability requirements

- Validate file type by decoding content, enforce file byte and pixel limits, and reject malformed files.
- Avoid rendering untrusted SVG/HTML content. Keep uploaded files outside application source directories.
- Separate customer data and avoid training on customer images without explicit permission.
- Never silently classify failed alignment as pass; return warning and require review.
- Store reproducibility metadata and calibrated thresholds by board design.
