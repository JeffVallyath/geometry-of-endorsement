"""The supported scientific CPU suite never requires a desktop GUI."""
import os

# Process-local test configuration; do not edit the user's matplotlibrc.
# Keep this stdlib-only so the fresh-study NumPy-only checks stay lightweight.
os.environ['MPLBACKEND'] = 'Agg'
