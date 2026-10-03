from statistics import median


def summarize(rows: list[dict]) -> dict:
    metrics = [x["metrics"] for x in rows if x.get("metrics")]
    timings = {}
    names = sorted({name for row in rows for name in row.get("timing_ms", {})})
    for name in names:
        values = [row["timing_ms"][name] for row in rows if name in row.get("timing_ms", {})]
        timings[name] = {
            "median_ms": median(values),
            "total_ms": sum(values),
        }
    out = {"queries": len(rows), "timing": timings}
    if metrics:
        out["retrieval"] = {
            key: sum(x[key] for x in metrics) / len(metrics)
            for key in ("recall","mrr","ndcg")
        }
    return out
