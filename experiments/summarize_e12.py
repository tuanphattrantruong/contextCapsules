#!/usr/bin/env python3
import json
import statistics
import sys

runs = [json.load(open(path)) for path in sys.argv[1:]]
med = lambda key: statistics.median(r[key] for r in runs)
out = {
    "experiment": "E12_service_backed_llamacpp",
    "runs": len(runs),
    "model": runs[0]["model"],
    "n_prompt": runs[0]["n_prompt"],
    "engine_blob_bytes": runs[0]["engine_blob_bytes"],
    "byte_exact_runs": sum(r["byte_exact"] for r in runs),
    "token_identical_runs": sum(r["generated_tokens_identical"] for r in runs),
    "hard_restart_recovery_runs": sum(r["post_restart_stat_ok"] for r in runs),
    "post_revoke_deny_runs": sum(r["post_revoke_denied"] for r in runs),
    "json_service_put_ms_median": med("json_service_put_ms"),
    "binary_service_put_ms_median": med("binary_service_put_ms"),
    "json_service_read_ms_p50_median": med("json_service_read_ms_p50"),
    "json_service_read_ms_p99_median": med("json_service_read_ms_p99"),
    "binary_service_read_ms_p50_median": med("binary_service_read_ms_p50"),
    "binary_service_read_ms_p99_median": med("binary_service_read_ms_p99"),
    "plain_file_read_ms_p50_median": med("plain_file_read_ms_p50"),
    "json_service_over_plain_p50_median": med("json_service_over_plain_p50"),
    "binary_service_over_plain_p50_median": med("binary_service_over_plain_p50"),
    "engine_state_set_ms_median": statistics.median(
        r["import"]["state_set_ms"] for r in runs),
    "pass": all(r["pass"] for r in runs),
    "scope": runs[0]["scope"],
}
print(json.dumps(out, indent=2))
