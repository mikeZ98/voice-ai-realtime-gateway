"""Deployment artifacts agree with the code they deploy.

The manifests, the Dockerfile and the runtime lock are parsed and cross-checked against
the gateway itself (ports, probe routes, preStop arguments, drain budgets, metric
names, settings), so a drift between them fails CI instead of a rollout.
"""

from __future__ import annotations

import base64
import re
import signal
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from voice_ai_realtime_gateway.deploy import app as app_module
from voice_ai_realtime_gateway.deploy import drain
from voice_ai_realtime_gateway.deploy.app import GatewaySettings, build_app, demo_pipeline
from voice_ai_realtime_gateway.deploy.lifecycle import _SHUTDOWN_GRACE_S
from voice_ai_realtime_gateway.production.observability import GatewayMetrics, SessionRegistry

pytestmark = pytest.mark.usefixtures("strict_loop")

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
K8S = DEPLOY / "k8s"
KEY = base64.urlsafe_b64encode(b"k" * 32).decode()

type Manifest = dict[str, Any]


@cache
def manifest(name: str) -> Manifest:
    document = yaml.safe_load((K8S / name).read_text())
    assert isinstance(document, dict)
    return document


def pod() -> Manifest:
    spec: Manifest = manifest("deployment.yaml")["spec"]["template"]["spec"]
    return spec


def container() -> Manifest:
    (only,) = pod()["containers"]
    assert isinstance(only, dict)
    return only


def settings() -> GatewaySettings:
    return GatewaySettings.from_env(
        {**manifest("configmap.yaml")["data"], "GATEWAY_KEYS": f"k1={KEY}"}
    )


@cache
def dockerfile() -> str:
    return (DEPLOY / "Dockerfile").read_text()


def instruction(name: str, text: str) -> list[str]:
    joined = text.replace("\\\n", " ")
    return [line.split(None, 1)[1] for line in joined.splitlines() if line.startswith(f"{name} ")]


def runtime_stage() -> str:
    return dockerfile().rsplit("\nFROM ", 1)[1]


def cmd() -> list[str]:
    (entry,) = instruction("ENTRYPOINT", runtime_stage())
    (args,) = instruction("CMD", runtime_stage())
    return [*yaml.safe_load(entry), *yaml.safe_load(args)]


def flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


# --------------------------------------------------------------------------- kustomize


def test_kustomization_lists_existing_files_and_exactly_one_autoscaler() -> None:
    resources = manifest("kustomization.yaml")["resources"]
    assert all((K8S / name).is_file() for name in resources)
    autoscalers = {"keda-scaledobject.yaml", "hpa.yaml"} & set(resources)
    assert len(autoscalers) == 1, "KEDA and an HPA must never fight over one Deployment"
    shipped = {path.name for path in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert shipped - set(resources) == {"hpa.yaml"}, "only the alternative autoscaler is unlisted"


def test_selectors_match_the_pod_labels() -> None:
    deployment = manifest("deployment.yaml")
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    selectors = [
        deployment["spec"]["selector"]["matchLabels"],
        manifest("service.yaml")["spec"]["selector"],
        manifest("pdb.yaml")["spec"]["selector"]["matchLabels"],
        *(c["labelSelector"]["matchLabels"] for c in pod()["topologySpreadConstraints"]),
    ]
    assert all(selector.items() <= labels.items() for selector in selectors)
    targets = [
        manifest("keda-scaledobject.yaml")["spec"]["scaleTargetRef"]["name"],
        manifest("hpa.yaml")["spec"]["scaleTargetRef"]["name"],
    ]
    assert targets == [deployment["metadata"]["name"]] * 2


# --------------------------------------------------------------------------- pod security


def test_pod_runs_non_root_on_a_read_only_filesystem() -> None:
    security, hardened = pod()["securityContext"], container()["securityContext"]
    assert security["runAsNonRoot"] is True
    (user,) = instruction("USER", runtime_stage())
    assert user == f"{security['runAsUser']}:{security['runAsGroup']}" == "10001:10001"
    assert security["seccompProfile"]["type"] == "RuntimeDefault"
    assert hardened["readOnlyRootFilesystem"] is True
    assert hardened["allowPrivilegeEscalation"] is False
    assert hardened["capabilities"]["drop"] == ["ALL"]
    assert pod()["automountServiceAccountToken"] is False
    assert "PYTHONDONTWRITEBYTECODE=1" in runtime_stage(), "nothing is written at runtime"


def test_rollouts_never_reduce_capacity() -> None:
    strategy = manifest("deployment.yaml")["spec"]["strategy"]
    assert strategy["rollingUpdate"]["maxUnavailable"] == 0
    assert manifest("pdb.yaml")["spec"]["maxUnavailable"] == 1


# --------------------------------------------------------------------------- ports & probes


def test_ports_agree_across_dockerfile_deployment_service_and_hook() -> None:
    (port,) = container()["ports"]
    argv = cmd()
    assert int(flag(argv, "--port")) == port["containerPort"] == 8080
    assert instruction("EXPOSE", runtime_stage()) == [str(port["containerPort"])]
    assert manifest("service.yaml")["spec"]["ports"][0]["targetPort"] == port["name"]
    hook = container()["lifecycle"]["preStop"]["exec"]["command"]
    assert flag(hook, "--url") == f"http://127.0.0.1:{port['containerPort']}/drainz"
    annotations = manifest("deployment.yaml")["spec"]["template"]["metadata"]["annotations"]
    assert int(annotations["prometheus.io/port"]) == port["containerPort"]


@pytest.mark.parametrize(
    ("probe", "expected"),
    [("startupProbe", 200), ("livenessProbe", 200), ("readinessProbe", 200)],
)
async def test_probes_hit_routes_the_app_serves(probe: str, expected: int) -> None:
    path = container()[probe]["httpGet"]["path"]
    gateway = build_app(settings(), demo_pipeline)
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await gateway({"type": "http", "method": "GET", "path": path}, receive, send)
    assert sent[0]["status"] == expected


def test_metrics_annotation_points_at_the_exposition_route() -> None:
    annotations = manifest("deployment.yaml")["spec"]["template"]["metadata"]["annotations"]
    assert annotations["prometheus.io/path"] == "/metrics"


# --------------------------------------------------------------------------- drain budget


def test_prestop_hook_is_a_valid_drain_invocation() -> None:
    hook = container()["lifecycle"]["preStop"]["exec"]["command"]
    assert hook[:3] == ["python", "-m", drain.__name__]
    calls: list[tuple[int, int]] = []
    status = drain.main(
        hook[3:], kill=lambda pid, sig: calls.append((pid, sig)), probe=lambda _url: 200
    )
    assert status == 0
    assert calls == [(1, signal.SIGUSR1)], "SIGUSR1 to the uvicorn process (PID 1)"


def test_drain_budgets_nest_inside_the_termination_grace_period() -> None:
    config = settings()
    hook = container()["lifecycle"]["preStop"]["exec"]["command"]
    hook_timeout = float(flag(hook, "--timeout"))
    full_drain = config.drain_timeout_s + _SHUTDOWN_GRACE_S + config.final_scrape_grace_s
    assert hook_timeout >= full_drain, "the hook must outwait a worst-case drain"
    uvicorn_grace = float(flag(cmd(), "--timeout-graceful-shutdown"))
    assert pod()["terminationGracePeriodSeconds"] > hook_timeout + uvicorn_grace, (
        "no SIGKILL mid-drain"
    )


def test_configmap_is_a_valid_gateway_configuration() -> None:
    config = settings()
    assert config.websocket_path == "/v1/voice"
    assert "default" in config.tenants
    tenant_cap = config.tenants["default"].max_concurrent_sessions
    threshold = float(
        manifest("keda-scaledobject.yaml")["spec"]["triggers"][0]["metadata"]["threshold"]
    )
    assert threshold < tenant_cap, "scale out before a pod starts refusing calls"


# --------------------------------------------------------------------------- autoscaling


def test_autoscalers_track_a_metric_the_gateway_exports() -> None:
    exported = set(
        re.findall(r"^# TYPE (\S+) ", GatewayMetrics(SessionRegistry()).render(), re.MULTILINE)
    )
    query = manifest("keda-scaledobject.yaml")["spec"]["triggers"][0]["metadata"]["query"]
    hpa_metric = manifest("hpa.yaml")["spec"]["metrics"][0]["pods"]["metric"]["name"]
    assert hpa_metric == "voice_gateway_active_sessions_count"
    assert hpa_metric in exported
    assert hpa_metric in query
    for scaler in (manifest("keda-scaledobject.yaml")["spec"], manifest("hpa.yaml")["spec"]):
        assert scaler.get("minReplicaCount", scaler.get("minReplicas")) >= 3, "PDB needs headroom"


def test_scale_down_is_slower_than_a_call_drain() -> None:
    behaviours = [
        manifest("keda-scaledobject.yaml")["spec"]["advanced"]["horizontalPodAutoscalerConfig"][
            "behavior"
        ],
        manifest("hpa.yaml")["spec"]["behavior"],
    ]
    for behaviour in behaviours:
        assert behaviour["scaleDown"]["stabilizationWindowSeconds"] >= 300
        assert behaviour["scaleDown"]["policies"] == [
            {"type": "Pods", "value": 1, "periodSeconds": 120}
        ]


# --------------------------------------------------------------------------- image


def test_dockerfile_is_multi_stage_pinned_and_cache_free() -> None:
    text = dockerfile()
    stages = re.findall(r"^FROM (\S+)(?: AS (\w+))?", text, re.MULTILINE)
    assert [name for _, name in stages] == ["uv", "build", "runtime"]
    digest = r"@sha256:[0-9a-f]{64}"
    (base,) = re.findall(r"^ARG PYTHON_IMAGE=(\S+)$", text, re.MULTILINE)
    assert re.fullmatch(rf"python:3\.12\.\d+-slim-\w+{digest}", base), "patch-level, by digest"
    (uv,) = re.findall(r"^ARG UV_IMAGE=(\S+)$", text, re.MULTILINE)
    assert re.fullmatch(rf"ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+{digest}", uv)
    assert stages[0][0] == "${UV_IMAGE}"
    assert re.match(rf"# syntax=docker/dockerfile:\d+\.\d+{digest}\n", text), "pinned frontend"
    assert "latest" not in text
    for setting in ("PIP_NO_CACHE_DIR=1", "UV_NO_CACHE=1", "PYTHONDONTWRITEBYTECODE=1"):
        assert setting in text
    assert "--require-hashes" in text
    assert "--invalidation-mode checked-hash" in text, "mtime-free, reproducible bytecode"
    # Bit-for-bit reproducibility: build-time epoch, fixed hash seed, no wall-clock
    # artifacts (uv_cache.json, useradd's /etc/shadow stamp), clamped timestamps.
    assert "ARG SOURCE_DATE_EPOCH=0" in text
    assert "PYTHONHASHSEED=0" in text
    assert "uv_cache.json" in text
    instructions = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("useradd" in line for line in instructions), "user written into passwd"
    assert '-newermt "@${SOURCE_DATE_EPOCH}"' in text
    assert "COPY --from=build --chown=root:root /opt/gateway /opt/gateway" in runtime_stage()
    assert "COPY src" not in runtime_stage(), "no build context in the runtime image"


def test_image_entrypoint_serves_the_deploy_factory() -> None:
    argv = cmd()
    assert argv[:3] == ["python", "-m", "uvicorn"]
    module, _, attribute = flag(argv, "--factory").partition(":")
    assert module == app_module.__name__
    assert callable(getattr(app_module, attribute))
    assert flag(argv, "--lifespan") == "on", "lifespan installs the drain signal handler"
    assert flag(argv, "--ws") == "websockets"


def test_runtime_lock_is_fully_pinned_with_hashes() -> None:
    text = (DEPLOY / "requirements-runtime.txt").read_text()
    requirements = [line for line in text.splitlines() if line and not line.startswith(("#", " "))]
    assert {r.split("==")[0] for r in requirements} >= {"uvicorn", "websockets"}
    for requirement in requirements:
        assert re.fullmatch(r"[a-z0-9-]+==[\w.]+ \\", requirement), requirement
    blocks = re.split(
        r"\n(?=\S)", "\n".join(row for row in text.splitlines() if not row.startswith("#"))
    )
    assert all("--hash=sha256:" in block for block in blocks if block.strip())


def test_dockerignore_ships_only_the_build_inputs() -> None:
    ignore = (DEPLOY.parent / ".dockerignore").read_text().splitlines()
    assert ignore[ignore.index("*") + 1 :][:6] == [
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!src/",
        "!deploy/requirements-runtime.txt",
        "**/__pycache__",
    ]
