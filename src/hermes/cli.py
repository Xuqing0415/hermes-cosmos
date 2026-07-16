"""
Hermes CLI - Command line interface for Hermes Cosmos
"""

import asyncio
from typing import Optional

import click
import httpx
import structlog

from hermes import __version__

logger = structlog.get_logger()


@click.group()
@click.version_option(version=__version__)
@click.option("--api-url", default="http://localhost:8080", help="Hermes API URL")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, api_url: str, debug: bool) -> None:
    """Hermes Cosmos CLI - Global AI Training Scheduling System"""
    ctx.ensure_object(dict)
    ctx.obj["api_url"] = api_url
    ctx.obj["debug"] = debug


@cli.group()
@click.pass_context
def jobs(ctx: click.Context) -> None:
    """Manage training jobs"""
    pass


@jobs.command("list")
@click.option("--tenant-id", help="Filter by tenant ID")
@click.option("--status", help="Filter by status")
@click.option("--page", default=1, help="Page number")
@click.option("--page-size", default=20, help="Page size")
@click.pass_context
def list_jobs(
    ctx: click.Context,
    tenant_id: Optional[str],
    status: Optional[str],
    page: int,
    page_size: int,
) -> None:
    """List all training jobs"""
    api_url = ctx.obj["api_url"]

    async def _list_jobs() -> None:
        params = {"page": page, "page_size": page_size}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if status:
            params["status"] = status

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/v1/jobs", params=params)
            response.raise_for_status()
            data = response.json()

        click.echo(f"Total jobs: {data['total']}")
        for job in data["jobs"]:
            click.echo(f"  - {job['id']}: {job['name']} ({job['status']})")

    asyncio.run(_list_jobs())


@jobs.command("submit")
@click.argument("name")
@click.option("--tenant-id", required=True, help="Tenant ID")
@click.option("--user-id", required=True, help="User ID")
@click.option("--gpu-count", default=1, help="Number of GPUs")
@click.option("--gpu-type", default="nvidia-h100", help="GPU type")
@click.option("--image", required=True, help="Docker image")
@click.option("--command", help="Command to run")
@click.pass_context
def submit_job(
    ctx: click.Context,
    name: str,
    tenant_id: str,
    user_id: str,
    gpu_count: int,
    gpu_type: str,
    image: str,
    command: Optional[str],
) -> None:
    """Submit a new training job"""
    api_url = ctx.obj["api_url"]

    payload = {
        "name": name,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "image": image,
        "requirements": {
            "gpu_count": gpu_count,
            "gpu_type": gpu_type,
        },
    }

    if command:
        payload["command"] = command

    async def _submit_job() -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{api_url}/v1/jobs", json=payload)
            response.raise_for_status()
            data = response.json()

        click.echo(f"Job submitted: {data['job']['id']}")

    asyncio.run(_submit_job())


@jobs.command("get")
@click.argument("job-id")
@click.pass_context
def get_job(ctx: click.Context, job_id: str) -> None:
    """Get job details"""
    api_url = ctx.obj["api_url"]

    async def _get_job() -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/v1/jobs/{job_id}")
            response.raise_for_status()
            data = response.json()

        job = data["job"]
        click.echo(f"Job ID: {job['id']}")
        click.echo(f"Name: {job['name']}")
        click.echo(f"Status: {job['status']}")
        click.echo(f"Tenant: {job['tenant_id']}")
        click.echo(f"User: {job['user_id']}")

    asyncio.run(_get_job())


@jobs.command("cancel")
@click.argument("job-id")
@click.pass_context
def cancel_job(ctx: click.Context, job_id: str) -> None:
    """Cancel a job"""
    api_url = ctx.obj["api_url"]

    async def _cancel_job() -> None:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{api_url}/v1/jobs/{job_id}")
            response.raise_for_status()

        click.echo(f"Job {job_id} cancelled")

    asyncio.run(_cancel_job())


@cli.group()
@click.pass_context
def checkpoints(ctx: click.Context) -> None:
    """Manage checkpoints"""
    pass


@checkpoints.command("list")
@click.option("--job-id", help="Filter by job ID")
@click.pass_context
def list_checkpoints(ctx: click.Context, job_id: Optional[str]) -> None:
    """List checkpoints"""
    api_url = ctx.obj["api_url"]

    async def _list_checkpoints() -> None:
        params = {}
        if job_id:
            params["job_id"] = job_id

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/v1/checkpoints", params=params)
            response.raise_for_status()
            data = response.json()

        click.echo(f"Total checkpoints: {data['total']}")
        for cp in data["checkpoints"]:
            click.echo(f"  - {cp['id']}: Job {cp['job_id']} ({cp['state']})")

    asyncio.run(_list_checkpoints())


@cli.group()
@click.pass_context
def resources(ctx: click.Context) -> None:
    """Manage resources"""
    pass


@resources.command("list")
@click.option("--region", help="Filter by region")
@click.pass_context
def list_resources(ctx: click.Context, region: Optional[str]) -> None:
    """List resources"""
    api_url = ctx.obj["api_url"]

    async def _list_resources() -> None:
        params = {}
        if region:
            params["region"] = region

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/v1/resources", params=params)
            response.raise_for_status()
            data = response.json()

        click.echo(f"Total resources: {data['total']}")
        for res in data["resources"]:
            click.echo(f"  - {res['id']}: {res['name']} ({res['type']})")

    asyncio.run(_list_resources())


@resources.command("summary")
@click.pass_context
def resource_summary(ctx: click.Context) -> None:
    """Get cluster resource summary"""
    api_url = ctx.obj["api_url"]

    async def _resource_summary() -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/v1/resources/cluster/summary")
            response.raise_for_status()
            data = response.json()

        click.echo("Cluster Summary:")
        click.echo(f"  Total GPUs: {data['total_gpus']}")
        click.echo(f"  Available GPUs: {data['available_gpus']}")
        click.echo(f"  Total Jobs: {data['total_jobs']}")
        click.echo(f"  Running Jobs: {data['running_jobs']}")
        click.echo(f"  Pending Jobs: {data['pending_jobs']}")
        click.echo("\nBy Region:")
        for region, count in data["regions"].items():
            click.echo(f"  {region}: {count} GPUs")

    asyncio.run(_resource_summary())


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check system status"""
    api_url = ctx.obj["api_url"]

    async def _status() -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/v1/health")
            response.raise_for_status()
            data = response.json()

        click.echo(f"Status: {data['status']}")
        click.echo(f"Version: {data['version']}")
        click.echo(f"Timestamp: {data['timestamp']}")
        click.echo("\nComponents:")
        for component, status in data["components"].items():
            click.echo(f"  {component}: {status}")

    asyncio.run(_status())


@cli.group()
@click.pass_context
def proof(ctx: click.Context) -> None:
    """Neural-symbolic proof generation commands"""
    pass


@proof.command("prove")
@click.argument("function-name")
@click.argument("module-file", type=click.File("r"))
@click.option("--solver", default="z3", help="Theorem prover solver")
@click.option("--timeout", type=int, default=30, help="Proof timeout in seconds")
@click.pass_context
def proof_prove(
    ctx: click.Context,
    function_name: str,
    module_file,
    solver: str,
    timeout: int,
) -> None:
    """Prove a function's correctness"""
    from hermes.neural_symbolic.proof_generator import ProofGenerator

    code = module_file.read()
    generator = ProofGenerator(solver=solver, timeout=timeout)
    result = generator.prove_and_generate_test(code, function_name)

    click.echo("=" * 60)
    click.echo("Neural-Symbolic Proof Results")
    click.echo("=" * 60)
    click.echo(f"  Proven:   {result['proven']}")
    click.echo(f"  Disproven: {result['disproven']}")
    click.echo(f"  Unknown:  {result['unknown']}")
    click.echo(f"  Duration: {result['duration']:.2f}s")

    if result["tests"]:
        click.echo("\nGenerated Tests:")
        click.echo("-" * 40)
        for test in result["tests"]:
            click.echo(f"\n  Inputs:   {test['inputs']}")
            click.echo(f"  Assertion: {test['assertion']}")

        click.echo("\nTest Code:")
        click.echo("-" * 40)
        click.echo(result["test_code"])


@proof.command("verify")
@click.argument("function-name")
@click.argument("module-file", type=click.File("r"))
@click.pass_context
def proof_verify(ctx: click.Context, function_name: str, module_file) -> None:
    """Verify division safety of a function"""
    from hermes.neural_symbolic.proof_generator import ProofGenerator

    code = module_file.read()
    generator = ProofGenerator()
    result = generator.verify_division_safety(code, function_name)

    click.echo("=" * 60)
    click.echo("Division Safety Verification")
    click.echo("=" * 60)
    click.echo(f"  Result: {result['result']}")
    click.echo(f"  Safe:   {result['safe']}")
    click.echo(f"  Proof Results: {result['proof_results']}")
    click.echo(f"  Generated Tests: {result['generated_tests']}")

    if result["test_code"]:
        click.echo("\nGenerated Test Code:")
        click.echo("-" * 40)
        click.echo(result["test_code"])


@proof.command("test")
@click.argument("function-name")
@click.argument("module-file", type=click.File("r"))
@click.option("--output", default="generated_tests.py", help="Output test file")
@click.pass_context
def proof_test(
    ctx: click.Context,
    function_name: str,
    module_file,
    output: str,
) -> None:
    """Generate tests from proof results"""
    from hermes.neural_symbolic.proof_generator import ProofGenerator

    code = module_file.read()
    generator = ProofGenerator()
    result = generator.prove_and_generate_test(code, function_name)

    if result["test_code"]:
        with open(output, "w") as f:
            f.write(result["test_code"])
        click.echo(f"Test file generated: {output}")
        click.echo(f"Generated {len(result['tests'])} test(s)")
    else:
        click.echo("No tests generated - all proofs succeeded!")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
