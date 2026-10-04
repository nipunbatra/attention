# Reproduce the recorded outputs

Extract recording-kit.zip, then open notebooks/record_grounded_vision.ipynb.
The replay cells inspect the included data and images. The optional recording
cells download about 1 GB of public, revision-pinned model weights and replace
the recorded JSON with a new CPU run. All detections are retained.

From the extracted folder, in a separate Python 3.12 environment:

```sh
python -m pip install -r requirements-models.txt
python src/prepare_models.py
python src/record_models.py
```

The recording scripts were executed; the convenience notebook was not executed
end-to-end. The requirements record the environment used; a fresh installation
was not tested. Re-recording does not modify the published HTML snapshots.
