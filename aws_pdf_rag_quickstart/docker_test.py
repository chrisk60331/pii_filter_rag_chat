#!/usr/bin/env python3
"""
Docker test script for Streamlit RAG application.
Tests that all components are working correctly in the containerized environment.
"""

import os
import sys
import time
import requests
import subprocess
from typing import Dict, List, Tuple

def test_streamlit_health() -> Tuple[bool, str]:
    """Test that Streamlit app is healthy and responding"""
    try:
        # Test the Streamlit health endpoint
        response = requests.get("http://localhost:8501/healthz", timeout=10)
        if response.status_code == 200:
            return True, "Streamlit app is healthy"
        else:
            return False, f"Streamlit health check failed: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Streamlit connection failed: {str(e)}"

def test_opensearch_health() -> Tuple[bool, str]:
    """Test that OpenSearch is healthy and responding"""
    try:
        response = requests.get("http://localhost:9200/_cluster/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            return True, f"OpenSearch is healthy (status: {status})"
        else:
            return False, f"OpenSearch health check failed: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"OpenSearch connection failed: {str(e)}"

def test_localstack_health() -> Tuple[bool, str]:
    """Test that LocalStack is healthy and responding"""
    try:
        response = requests.get("http://localhost:4566/health", timeout=10)
        if response.status_code == 200:
            return True, "LocalStack is healthy"
        else:
            return False, f"LocalStack health check failed: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"LocalStack connection failed: {str(e)}"

def test_docker_services() -> Dict[str, Tuple[bool, str]]:
    """Test all Docker services"""
    services = {
        "Streamlit": test_streamlit_health,
        "OpenSearch": test_opensearch_health,
        "LocalStack": test_localstack_health,
    }
    
    results = {}
    for service_name, test_func in services.items():
        print(f"Testing {service_name}...")
        try:
            success, message = test_func()
            results[service_name] = (success, message)
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status}: {message}")
        except Exception as e:
            results[service_name] = (False, f"Test error: {str(e)}")
            print(f"  ❌ FAIL: Test error: {str(e)}")
    
    return results

def check_docker_compose() -> bool:
    """Check if docker-compose is available"""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def main():
    """Main test function"""
    print("🐳 Docker Streamlit RAG Application Test")
    print("=" * 50)
    
    # Check if docker-compose is available
    if not check_docker_compose():
        print("❌ docker-compose not found. Please install docker-compose.")
        sys.exit(1)
    
    print("✅ docker-compose found")
    
    # Wait a bit for services to be ready
    print("\n⏳ Waiting for services to be ready...")
    time.sleep(5)
    
    # Test all services
    print("\n🔍 Testing Docker services...")
    results = test_docker_services()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    all_passed = True
    for service, (success, message) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {service}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Streamlit app is ready to use.")
        print("📱 Access the app at: http://localhost:8501")
        print("🔍 OpenSearch Dashboard: http://localhost:5601")
        print("☁️  LocalStack: http://localhost:4566")
    else:
        print("\n❌ Some tests failed. Check the services and try again.")
        
        # Provide troubleshooting tips
        print("\n🔧 Troubleshooting tips:")
        print("  1. Make sure all containers are running: docker-compose ps")
        print("  2. Check container logs: docker-compose logs [service-name]")
        print("  3. Restart services: docker-compose restart")
        print("  4. Rebuild and restart: docker-compose up --build")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 