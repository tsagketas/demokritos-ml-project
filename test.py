print("hello welcome")

# Test imports for each library
print("\nTesting installed libraries:")

try:
    import sklearn
    print("✓ sklearn (scikit-learn) is installed - version:", sklearn.__version__)
except ImportError:
    print("✗ sklearn (scikit-learn) is NOT installed")

try:
    import pandas
    print("✓ pandas is installed - version:", pandas.__version__)
except ImportError:
    print("✗ pandas is NOT installed")

try:
    import numpy
    print("✓ numpy is installed - version:", numpy.__version__)
except ImportError:
    print("✗ numpy is NOT installed")

# try:
#     import torch
#     print("✓ torch is installed - version:", torch.__version__)
# except ImportError:
#     print("✗ torch is NOT installed (expected if commented out in requirements.txt)")

print("\nAll tests completed!")

