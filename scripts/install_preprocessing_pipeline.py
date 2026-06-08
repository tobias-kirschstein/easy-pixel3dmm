"""
Cross-platform installation script for the preprocessing pipeline.
Equivalent to install_preprocessing_pipeline.sh.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    import requests


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, shell=(sys.platform == "win32"))


def gdown_download(file_id, output):
    run(f"gdown --id {file_id} -O {output}")


def install_facer(preprocessing_dir):
    facer_dir = preprocessing_dir / "facer"
    if not facer_dir.exists():
        run(f"git clone git@github.com:FacePerceiver/facer.git", cwd=preprocessing_dir)

    replacement = preprocessing_dir / "replacement_code"
    shutil.copy(replacement / "farl.py", facer_dir / "facer" / "face_parsing" / "farl.py")
    shutil.copy(replacement / "facer_transform.py", facer_dir / "facer" / "transform.py")
    run(f"{sys.executable} -m pip install -e .", cwd=facer_dir)


def install_mica(preprocessing_dir):
    mica_dir = preprocessing_dir / "MICA"
    if not mica_dir.exists():
        run("git clone git@github.com:Zielon/MICA.git", cwd=preprocessing_dir)

    replacement = preprocessing_dir / "replacement_code"
    shutil.copy(replacement / "mica_demo.py", mica_dir / "demo.py")
    shutil.copy(replacement / "mica.py", mica_dir / "micalib" / "models" / "mica.py")

    _run_mica_install(mica_dir)


def _run_mica_install(mica_dir):
    print("\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction.")
    username = input("Username (FLAME): ")
    password = input("Password (FLAME): ")

    def wget_post(url, output):
        """Download a file using a form POST request (equivalent to wget --post-data --no-check-certificate)."""
        output_path = mica_dir / output
        print(f"\nDownloading to {output_path} ...")
        response = requests.post(
            url,
            data={"username": username, "password": password},
            verify=False,
            stream=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(
                f"Server returned an HTML page instead of a file — credentials may be incorrect.\n"
                f"URL: {url}"
            )
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    def unzip(archive, dest_dir):
        dest = mica_dir / dest_dir
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(mica_dir / archive) as zf:
            zf.extractall(dest)

    print("\nDownloading FLAME...")
    (mica_dir / "data").mkdir(parents=True, exist_ok=True)
    wget_post(
        "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1",
        "FLAME2020.zip",
    )
    unzip("FLAME2020.zip", "data")
    (mica_dir / "FLAME2020.zip").unlink()

    wget_post(
        "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1",
        "FLAME2023.zip",
    )
    unzip("FLAME2023.zip", "data")
    (mica_dir / "FLAME2023.zip").unlink()

    # Ensure gdown is available
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown...")
        run(f"{sys.executable} -m pip install gdown")

    print("\nDownloading MICA weights...")
    pretrained_dir = mica_dir / "data" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    gdown_download("1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt", str(pretrained_dir / "mica.tar"))

    print("\nDownloading insightface models...")
    insightface_dir = Path.home() / ".insightface" / "models"
    insightface_dir.mkdir(parents=True, exist_ok=True)

    antelopev2_dir = insightface_dir / "antelopev2"
    if not antelopev2_dir.exists():
        zip_path = insightface_dir / "antelopev2.zip"
        gdown_download("16PWKI_RjjbE4_kqpElG-YFqe8FpXjads", str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(antelopev2_dir)

    buffalo_l_dir = insightface_dir / "buffalo_l"
    if not buffalo_l_dir.exists():
        zip_path = insightface_dir / "buffalo_l.zip"
        gdown_download("1navJMy0DTr1_DHjLWu1i48owCPvXWfYc", str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(buffalo_l_dir)

    print("\nMICA installation has finished!")


def install_pipnet(preprocessing_dir):
    pipnet_dir = preprocessing_dir / "PIPNet"
    if not pipnet_dir.exists():
        run("git clone https://github.com/jhb86253817/PIPNet.git", cwd=preprocessing_dir)

    # Build FaceBoxesV2 C extensions (replaces: sh make.sh -> python3 build.py build_ext --inplace)
    faceboxes_utils = pipnet_dir / "FaceBoxesV2" / "utils"
    subprocess.run(
        [sys.executable, "build.py", "build_ext", "--inplace"],
        cwd=faceboxes_utils,
        check=True,
    )

    # Create snapshot directories and download weights
    snapshot_dir = pipnet_dir / "snapshots" / "WFLW" / "pip_32_16_60_r18_l2_l1_10_1_nb10"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    gdown_download("1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C", str(snapshot_dir / "epoch59.pth"))


def download_pretrained_weights(repo_root):
    weights_dir = repo_root / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    gdown_download("1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY", str(weights_dir / "uv.ckpt"))
    gdown_download("1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD", str(weights_dir / "normals.ckpt"))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    preprocessing_dir = repo_root / "src" / "pixel3dmm" / "preprocessing"

    print("=== Installing facer ===")
    install_facer(preprocessing_dir)

    print("\n=== Installing MICA ===")
    install_mica(preprocessing_dir)

    print("\n=== Installing PIPNet ===")
    install_pipnet(preprocessing_dir)

    print("\n=== Downloading pretrained weights ===")
    download_pretrained_weights(repo_root)

    print("\nAll done!")
