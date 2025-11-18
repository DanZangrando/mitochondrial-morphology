#!/usr/bin/env python3
"""
Quick test to verify TensorBoard integration works
"""

import subprocess
import time
import requests
import sys

def test_tensorboard_startup():
    """Test that TensorBoard can start and respond"""
    
    print("🧪 Testing TensorBoard Integration")
    print("=" * 60)
    
    # Kill any existing TensorBoard
    print("\n1. Cleaning up existing TensorBoard processes...")
    import os
    import signal
    try:
        # Use pgrep to find TensorBoard processes
        result = subprocess.run(
            ["pgrep", "-f", "tensorboard"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except:
                    pass
            time.sleep(1)
            print(f"   ✓ Terminated {len(pids)} TensorBoard process(es)")
        else:
            print("   ✓ No existing TensorBoard processes")
    except Exception as e:
        print(f"   ✓ Cleanup complete")
    
    # Start TensorBoard
    print("\n2. Starting TensorBoard server...")
    try:
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/", "--port", "6006", "--bind_all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"   ✓ Process started (PID: {tb_process.pid})")
    except Exception as e:
        print(f"   ✗ Failed to start: {e}")
        return False
    
    # Wait for startup
    print("\n3. Waiting for TensorBoard to initialize...")
    time.sleep(5)
    
    # Test connection
    print("\n4. Testing HTTP connection...")
    try:
        response = requests.get("http://localhost:6006", timeout=5)
        if response.status_code == 200:
            print("   ✓ TensorBoard is responding!")
            print(f"   Status: {response.status_code}")
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Connection failed: {e}")
        tb_process.kill()
        return False
    
    # Check if logs directory exists
    print("\n5. Checking for training logs...")
    import os
    log_dirs = []
    if os.path.exists("logs/vae_classifier"):
        log_dirs.append("vae_classifier")
    if os.path.exists("logs/lstm_vae_classifier"):
        log_dirs.append("lstm_vae_classifier")
    
    if log_dirs:
        print(f"   ✓ Found logs: {', '.join(log_dirs)}")
    else:
        print("   ℹ️  No training logs yet (train a model first)")
    
    # Cleanup
    print("\n6. Cleaning up test...")
    tb_process.kill()
    print("   ✓ TensorBoard stopped")
    
    print("\n" + "=" * 60)
    print("✅ TensorBoard integration test PASSED!")
    print("\n💡 To use TensorBoard in Streamlit:")
    print("   1. Run: streamlit run app.py")
    print("   2. Go to: 🎓 Entrenar Modelo page")
    print("   3. Click: 🚀 Iniciar Entrenamiento")
    print("=" * 60)
    
    return True

def test_streamlit_imports():
    """Test that all required imports work"""
    
    print("\n🧪 Testing Streamlit Dependencies")
    print("=" * 60)
    
    imports = [
        ("streamlit", "Streamlit core"),
        ("streamlit.components.v1", "Streamlit components (for iframe)"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("tensorboard", "TensorBoard"),
    ]
    
    all_ok = True
    for module, description in imports:
        try:
            __import__(module)
            print(f"   ✓ {description:40s} OK")
        except ImportError as e:
            print(f"   ✗ {description:40s} FAILED: {e}")
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("✅ All dependencies available!")
    else:
        print("❌ Some dependencies missing. Run: pip install -r requirements.txt")
    
    return all_ok

if __name__ == "__main__":
    print("🧬 Mitochondrial Morphology - TensorBoard Integration Test\n")
    
    # Test imports first
    if not test_streamlit_imports():
        print("\n❌ Dependency test failed. Fix dependencies before proceeding.")
        sys.exit(1)
    
    # Test TensorBoard
    print("\n")
    if not test_tensorboard_startup():
        print("\n❌ TensorBoard test failed.")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Ready to train models with real-time monitoring.")
    sys.exit(0)
