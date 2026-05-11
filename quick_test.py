"""
Simple test script to verify the system is working
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("="*80)
print("HERMES COSMOS - QUICK VERIFICATION TEST")
print("="*80)
print()

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from hermes.core import config, models, exceptions
        print("✓ hermes.core imported")
        
        from hermes.core.config import Region, GPUType
        print("✓ hermes.core.config imported")
        
        from hermes.core.models import Job, JobStatus, JobPriority
        print("✓ hermes.core.models imported")
        
        from hermes.core.exceptions import HermesError
        print("✓ hermes.core.exceptions imported")
        
        from hermes.core.store import (
            get_job_queue,
            get_checkpoint_store,
            get_resource_manager
        )
        print("✓ hermes.core.store imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_models():
    """Test that data models work correctly"""
    print("\nTesting data models...")
    
    try:
        from hermes.core.models import Job, JobStatus, JobPriority, JobRequirements, PlacementConstraints
        from hermes.core.config import Region, GPUType
        
        # Test job creation
        job = Job(
            name="quick-test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            requirements=JobRequirements(gpu_count=8),
            constraints=PlacementConstraints(regions=[Region.US_EAST]),
            image="pytorch/pytorch:latest"
        )
        
        print(f"✓ Job created: {job.name}")
        print(f"  - ID: {job.id}")
        print(f"  - Status: {job.status}")
        print(f"  - GPU count: {job.requirements.gpu_count}")
        
        return True
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        return False


async def test_in_memory_store():
    """Test in-memory store functionality"""
    print("\nTesting in-memory store...")
    
    try:
        from hermes.core.models import Job, JobStatus, JobPriority, JobRequirements
        from hermes.core.config import Region
        from hermes.core.store import (
            get_job_queue,
            get_checkpoint_store,
            get_resource_manager
        )
        
        queue = get_job_queue()
        resource_manager = get_resource_manager()
        
        # Create a job
        job = Job(
            name="store-test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            requirements=JobRequirements(gpu_count=8),
            image="pytorch/pytorch:latest"
        )
        
        await queue.enqueue(job)
        print(f"✓ Job enqueued")
        
        # Check queue size
        all_jobs = await queue.get_all_jobs()
        print(f"✓ Queue has {len(all_jobs)} jobs")
        
        # Test resource manager
        cluster_state = await resource_manager.get_cluster_state()
        print(f"✓ Cluster state: {cluster_state['total_gpus']} GPUs total")
        print(f"  - Available: {cluster_state['available_gpus']}")
        print(f"  - Regions: {list(cluster_state['regions'].keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytest_importable():
    """Test that pytest is available and tests can be found"""
    print("\nTesting pytest configuration...")
    
    try:
        import pytest
        print("✓ pytest imported")
        
        tests_dir = project_root / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            print(f"✓ Found {len(test_files)} test files:")
            for tf in test_files:
                print(f"  - {tf.name}")
        
        return True
    except Exception as e:
        print(f"✗ Pytest test failed: {e}")
        return False


async def main():
    """Main test function"""
    passed = 0
    total = 0
    
    # Test imports
    total += 1
    if test_imports():
        passed += 1
    
    # Test models
    total += 1
    if test_models():
        passed += 1
    
    # Test store
    total += 1
    if await test_in_memory_store():
        passed += 1
    
    # Test pytest
    total += 1
    if test_pytest_importable():
        passed += 1
    
    print("\n" + "="*80)
    print(f"QUICK VERIFICATION RESULTS: {passed}/{total} passed")
    print("="*80)
    
    if passed == total:
        print("\n🎉 All quick checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -e \".[dev]\"")
        print("2. Run full tests: pytest tests/ -v")
        print("3. Start dev server: ./dev.sh scheduler")
        return 0
    else:
        print("\n⚠️ Some checks failed. Please resolve issues first.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
