#!/bin/sh
# Download the MedMNIST v2 archives used by this study into ./medmnist_data.
#
# The experiments also download on first use, so this script is only a
# convenience for preparing an offline machine.
set -e
D="$(cd "$(dirname "$0")" && pwd)/medmnist_data"
mkdir -p "$D"

BASE="https://zenodo.org/records/10519652/files"
for f in bloodmnist.npz; do
    if [ -f "$D/$f" ]; then
        echo "already present: $f"
    else
        echo "downloading $f"
        curl -L --fail -o "$D/$f" "$BASE/$f?download=1"
    fi
done

echo
echo "Expected checksums for bloodmnist.npz"
echo "  MD5     7053d0359d879ad8a5505303e11de1dc   (as published by MedMNIST v2)"
echo "  SHA-256 062023e186f537e26b3c21ea3b2614ddfc475e8a14825dfd20663bbb7e37bddc"
echo "Actual:"
md5 "$D"/*.npz 2>/dev/null || md5sum "$D"/*.npz
shasum -a 256 "$D"/*.npz
