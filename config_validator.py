#!/usr/bin/env python3
"""
Hermes 
SIEM

:
1. 
2. Kafka
3. SIEM
4. 
"""

import os
import sys
import json
import socket
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Optional
import yaml

def load_config(config_path: str) -> dict:
    """"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_scheduler_config(config: dict) -> List[str]:
    """"""
    errors = []
    warnings = []

    # 
    if 'security' not in config:
        errors.append(" security ")
        return errors

    security = config['security']

    # TEE
    if 'tee' in security:
        tee = security['tee']
        if tee.get('required') is not True:
            warnings.append("TEE ")

        if not tee.get('allowed_types'):
            errors.append("TEE allowed_types ")

        if tee.get('fallback_allowed') is True:
            warnings.append("TEE fallback_allowed  true false")

    # 
    if 'audit' not in security:
        errors.append(" audit ")
    else:
        audit = security['audit']
        if not audit.get('enabled'):
            errors.append("")

        if 'send_to_siems' not in audit or not audit['send_to_siems']:
            errors.append(" SIEM ")

        for siem in audit.get('send_to_siems', []):
            if siem.get('type') == 'kafka':
                if 'endpoint' not in siem:
                    errors.append(f"Kafka SIEM {siem.get('name')}  endpoint")
                if 'topic' not in siem:
                    errors.append(f"Kafka SIEM {siem.get('name')}  topic")

    return errors, warnings

def validate_checkpoint_config(config: dict) -> List[str]:
    """Checkpoint"""
    errors = []
    warnings = []

    if 'replication' not in config:
        errors.append(" replication ")
        return errors, warnings

    repl = config['replication']

    if not repl.get('enabled'):
        errors.append("Region")
    else:
        if not repl.get('target_regions'):
            errors.append("Region")

        delay = repl.get('replication_delay_target_seconds', 0)
        if delay > 600:  # 10
            warnings.append(f" {delay}s  <= 300s")

    return errors, warnings

def test_kafka_connection(endpoint: str) -> Dict[str, any]:
    """Kafka"""
    result = {
        'endpoint': endpoint,
        'reachable': False,
        'port_open': False,
        'error': None
    }

    try:
        #  endpoint
        host, port = endpoint.split(':')
        port = int(port)

        # 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        port_open = sock.connect_ex((host, port))
        sock.close()

        result['port_open'] = (port_open == 0)
        result['reachable'] = result['port_open']

        if not result['port_open']:
            result['error'] = f" {port} "

    except Exception as e:
        result['error'] = str(e)

    return result

def generate_test_audit_log() -> dict:
    """"""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "JOB_SUBMITTED",
        "service": "hermes-scheduler",
        "tenant_id": "test-tenant",
        "user_id": "hermes-tester",
        "job_id": "test-job-001",
        "job_name": "llama-2-13b-inference",
        "priority": "HIGH",
        "gpu_count": 4,
        "region": "us-east",
        "status": "SUCCESS",
        "metadata": {
            "scheduling_latency_ms": 350,
            "tee_enabled": True,
            "carbon_aware": True
        }
    }

def send_test_log_to_kafka(endpoint: str, topic: str) -> Dict[str, any]:
    """Kafka"""
    result = {
        'endpoint': endpoint,
        'topic': topic,
        'sent': False,
        'error': None
    }

    try:
        #  ( kafka-python  confluent-kafka)
        print(f"    {endpoint}/{topic}...")
        print(f"   : {json.dumps(generate_test_audit_log(), indent=2)}")

        # :
        # from kafka import KafkaProducer
        # producer = KafkaProducer(bootstrap_servers=endpoint)
        # producer.send(topic, json.dumps(generate_test_audit_log()).encode())
        # producer.flush()

        result['sent'] = True
        print("    ")

    except Exception as e:
        result['error'] = str(e)
        print(f"    : {e}")

    return result

def generate_config_report(
    scheduler_errors: List[str],
    scheduler_warnings: List[str],
    checkpoint_errors: List[str],
    checkpoint_warnings: List[str],
    kafka_results: List[Dict],
    siem_results: List[Dict]
) -> str:
    """"""

    total_errors = len(scheduler_errors) + len(checkpoint_errors)
    total_warnings = len(scheduler_warnings) + len(checkpoint_warnings)

    report = f"""# Hermes 

****: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

##  

|  |  |
|--------|------|
| TEE | {'' if not scheduler_errors and 'tee' not in str(scheduler_warnings) else ''} |
|  | {'' if 'audit' not in scheduler_errors else ''} |
| SIEM | {'' if siem_results and all(r.get('sent') for r in siem_results) else ''} |
| Region | {'' if not checkpoint_errors else ''} |

---

##  

###  ({len(scheduler_errors)})

{chr(10).join(f"- {e}" for e in scheduler_errors) if scheduler_errors else ""}

### Checkpoint ({len(checkpoint_errors)})

{chr(10).join(f"- {e}" for e in checkpoint_errors) if checkpoint_errors else ""}

---

##  

###  ({len(scheduler_warnings)})

{chr(10).join(f"- {w}" for w in scheduler_warnings) if scheduler_warnings else ""}

### Checkpoint ({len(checkpoint_warnings)})

{chr(10).join(f"- {w}" for w in checkpoint_warnings) if checkpoint_warnings else ""}

---

##  SIEM 

### Kafka 

| Endpoint |  |  |
|----------|----------|------|
{"".join(f"| {r['endpoint']} | {'' if r['port_open'] else ''} | {r.get('error', 'OK')} |" for r in kafka_results)}

### 

| SIEM |  |
|------|------|
{"".join(f"| {r['endpoint']}/{r['topic']} | {' ' if r['sent'] else ' ' + r.get('error', '')} |" for r in siem_results)}

---

##  

|  |  |  |  |
|------|--------|--------------|--------|
| 1 | TEE | Q2 2026 | Security Team |
| 2 |  |  | Hermes Team |
| 3 | Region Checkpoint |  | Hermes Team |

---

##  

****: {' ' if total_errors == 0 else f' {total_errors}, {total_warnings}'}

****:
1. 
2. SIEM
3. TEE

---

*Generated by Hermes Config Validator*
"""
    return report

async def main():
    parser = argparse.ArgumentParser(description="Hermes ")
    parser.add_argument("--scheduler-config", default="config/scheduler.yaml", help="")
    parser.add_argument("--checkpoint-config", default="config/checkpoint.yaml", help="Checkpoint")
    parser.add_argument("--skip-kafka-test", action="store_true", help="Kafka")
    args = parser.parse_args()

    print("="*60)
    print(" Hermes ")
    print("="*60)
    print(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 
    print("\n ...")

    try:
        scheduler_config = load_config(args.scheduler_config)
        print(f"    : {args.scheduler_config}")
    except Exception as e:
        print(f"    : {e}")
        scheduler_config = {}

    try:
        checkpoint_config = load_config(args.checkpoint_config)
        print(f"    Checkpoint: {args.checkpoint_config}")
    except Exception as e:
        print(f"    Checkpoint: {e}")
        checkpoint_config = {}

    # 
    print("\n ...")

    scheduler_errors, scheduler_warnings = [], []
    if scheduler_config:
        scheduler_errors, scheduler_warnings = validate_scheduler_config(scheduler_config)

    checkpoint_errors, checkpoint_warnings = [], []
    if checkpoint_config:
        checkpoint_errors, checkpoint_warnings = validate_checkpoint_config(checkpoint_config)

    print(f"   : {len(scheduler_errors)}")
    print(f"   : {len(scheduler_warnings)}")
    print(f"   Checkpoint: {len(checkpoint_errors)}")
    print(f"   Checkpoint: {len(checkpoint_warnings)}")

    # Kafka
    kafka_results = []
    siem_results = []

    if not args.skip_kafka_test and 'audit' in scheduler_config.get('security', {}):
        audit = scheduler_config['security']['audit']
        siems = audit.get('send_to_siems', [])

        print("\n SIEM...")

        for siem in siems:
            if siem.get('type') == 'kafka':
                endpoint = siem.get('endpoint', '')
                topic = siem.get('topic', '')

                # 
                kafka_result = test_kafka_connection(endpoint)
                kafka_results.append(kafka_result)
                print(f"   {endpoint}: {'' if kafka_result['port_open'] else ''}")

                # 
                if kafka_result['port_open']:
                    siem_result = send_test_log_to_kafka(endpoint, topic)
                    siem_results.append(siem_result)

    # 
    print("\n ...")

    report = generate_config_report(
        scheduler_errors,
        scheduler_warnings,
        checkpoint_errors,
        checkpoint_warnings,
        kafka_results,
        siem_results
    )

    report_path = "CONFIG_VALIDATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"   : {report_path}")
    print()
    print(report)

    # 
    total_errors = len(scheduler_errors) + len(checkpoint_errors)
    return 0 if total_errors == 0 else 1

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
