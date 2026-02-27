"""
fallback.py — Rule-Based Anomaly Detection Fallback

A standalone rule-based detector used when ML models are unavailable.
Can also be used as a baseline to compare ML model performance against.

Rules are based on domain knowledge of typical telecom network thresholds.

Usage:
    # As a module
    from serving.fallback import RuleBasedDetector
    detector = RuleBasedDetector()
    result = detector.predict({"latency_ms": 500, "packet_loss_pct": 12, ...})

    # As a standalone script for testing
    python serving/fallback.py
"""
from dataclasses import dataclass, field


@dataclass
class ThresholdRule:
    """A single threshold-based anomaly rule."""
    metric: str
    operator: str  # ">" or "<"
    threshold: float
    severity: str  # "warning" or "critical"
    description: str


# ─────────────────────────────────────────────
# Default Rules — based on telecom domain knowledge
# ─────────────────────────────────────────────
DEFAULT_RULES = [
    # Latency rules
    ThresholdRule("latency_ms", ">", 100,  "warning",  "High latency detected"),
    ThresholdRule("latency_ms", ">", 500,  "critical", "Critical latency spike"),

    # Packet loss rules
    ThresholdRule("packet_loss_pct", ">", 2.0,  "warning",  "Elevated packet loss"),
    ThresholdRule("packet_loss_pct", ">", 10.0, "critical", "Severe packet loss"),

    # CPU rules
    ThresholdRule("cpu_utilization", ">", 85,  "warning",  "High CPU utilization"),
    ThresholdRule("cpu_utilization", ">", 95,  "critical", "CPU saturation"),

    # Bandwidth rules
    ThresholdRule("bandwidth_mbps", "<", 50,  "warning",  "Low bandwidth"),
    ThresholdRule("bandwidth_mbps", "<", 10,  "critical", "Near-zero bandwidth"),

    # Error rate rules
    ThresholdRule("error_rate", ">", 20,  "warning",  "Elevated error rate"),
    ThresholdRule("error_rate", ">", 50,  "critical", "Critical error rate"),
]


@dataclass
class RuleBasedDetector:
    """
    Rule-based anomaly detector using configurable threshold rules.
    Used as a fallback when ML models are unavailable, and as a
    baseline for comparing ML model performance.
    """
    rules: list[ThresholdRule] = field(default_factory=lambda: DEFAULT_RULES.copy())

    def predict(self, metrics: dict) -> dict:
        """
        Check metrics against all rules.
        Returns prediction with triggered rules and severity.
        """
        triggered = []

        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                continue

            if rule.operator == ">" and value > rule.threshold:
                triggered.append({
                    "rule": f"{rule.metric} {rule.operator} {rule.threshold}",
                    "actual_value": value,
                    "severity": rule.severity,
                    "description": rule.description,
                })
            elif rule.operator == "<" and value < rule.threshold:
                triggered.append({
                    "rule": f"{rule.metric} {rule.operator} {rule.threshold}",
                    "actual_value": value,
                    "severity": rule.severity,
                    "description": rule.description,
                })

        is_anomaly = len(triggered) > 0
        has_critical = any(t["severity"] == "critical" for t in triggered)

        return {
            "is_anomaly": int(is_anomaly),
            "confidence": 1.0 if has_critical else (0.7 if is_anomaly else 0.0),
            "severity": "critical" if has_critical else ("warning" if is_anomaly else "normal"),
            "triggered_rules": triggered,
            "num_rules_triggered": len(triggered),
            "method": "rule-based",
        }

    def evaluate(self, data: list[dict], labels: list[int]) -> dict:
        """
        Evaluate rule-based detector against labeled data.
        Returns precision, recall, F1 for comparison with ML models.
        """
        tp = fp = fn = tn = 0

        for metrics, label in zip(data, labels):
            result = self.predict(metrics)
            pred = result["is_anomaly"]

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total": len(labels),
        }


# ─────────────────────────────────────────────
# CLI Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    detector = RuleBasedDetector()

    test_cases = [
        {"name": "Normal",   "latency_ms": 10, "packet_loss_pct": 0.02, "cpu_utilization": 40, "bandwidth_mbps": 500, "error_rate": 3},
        {"name": "Spike",    "latency_ms": 800, "packet_loss_pct": 15,  "cpu_utilization": 98, "bandwidth_mbps": 50,  "error_rate": 100},
        {"name": "Degraded", "latency_ms": 120, "packet_loss_pct": 3,   "cpu_utilization": 88, "bandwidth_mbps": 200, "error_rate": 25},
    ]

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Rule-Based Fallback Detector — Test            ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    for case in test_cases:
        name = case.pop("name")
        result = detector.predict(case)
        status = "🔴 ANOMALY" if result["is_anomaly"] else "🟢 NORMAL"
        print(f"  {name:<10} → {status} ({result['severity']})")
        for rule in result["triggered_rules"]:
            print(f"             ⚠️  {rule['description']} ({rule['rule']}, actual: {rule['actual_value']})")
        print()
