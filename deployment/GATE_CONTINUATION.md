# Flash Gate: authorized independent continuation

The 2026-10-07 continuation preserves the completed F37-Off result and every recorded F37-On turn. It executes only the remaining F37-On turns, then F38-Off and F38-On. It does not rebuild the album or retransmit the old uncertain request. Each derived result carries its parent run ID and SHA256; old runs remain immutable.

The user explicitly authorized keeping the single old uncertain reservation while completing the remaining work. This exception is supplied as a private `UncertainRequestAcknowledgement`, bound to the ledger and journal file identities, request and payload, original row hashes, and full reservation. It is not reconciliation. Ordinary clients still block on unknown requests. Any additional unknown request, changed acknowledgement, changed budget limit, or insufficient balance stops new submissions.

1. Freeze and deploy a new private source commit. Run `gate_validate.py` with `CAMROLL_GATE_VALIDATION_ROOT` set to the new continuation runtime.
2. Capture the specifically authorized request using `UncertainRequestAcknowledgement.capture`, supplying its known identity and payload hash. Store the resulting JSON only in the private runtime. Never infer authorization by enumerating all outstanding requests.
3. Run `gate_continuation_smoke.py --execute-paid --runtime <new-runtime> --acknowledgement <private-json>`. This checks both Flash models and Doubao using synthetic inputs and the scoped budget. It references the original frozen, successful end-to-end smoke rather than rebuilding the formal album.
4. Run `gate_continuation.py --execute-paid --runtime <new-runtime> --acknowledgement <private-json>`. The supervisor requires matching validation/smoke source manifests, unchanged substantive inference and scoring protocols, immutable parent records, the original snapshot, and the existing ¥600 ledger.
5. After the supervisor exits and restoration finishes, run `gate_continuation_export.py`. The read-only `gate_continuation_report.py` serves the new runtime and keeps previous suites available as history on port 8774. A port conflict is an error.

All execution uses independent request journals, query namespaces, claims and GPU1 lease windows. The shared GPU ownership registry and original service restoration rules remain in force. The resumed question receives only its own persisted conversation; completed tools and inherited scores are not rerun. The original failure turn remains an execution failure in the combined fixed-denominator result.

Execution completion and financial settlement are separate fields. A completed continuation can still have the specifically acknowledged reservation outstanding; reports must retain that distinction and split inherited versus new costs. No raw QA, media, vectors, journals, reservations or per-question results belong in Git.
