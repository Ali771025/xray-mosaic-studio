"""Streamable interoperable TIFF export with full metric sidecar and native float values."""
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np
import tifffile
from .geometry import GeometryError


def export_tiff(result_directory, destination):
    result = Path(result_directory).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise GeometryError('TIFF export destination must be new', 'FAILED')
    report = json.loads((result/'report.json').read_text(encoding='utf-8'))
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.xray-tiff-', dir=target.parent))
    try:
        for p in report['planes']:
            directory = work/p['directory']
            directory.mkdir()
            for name in ('intensity', 'coverage', 'source', 'metric_valid', 'sigma_mm'):
                original = np.load(result/p['directory']/(name+'.npy'), mmap_mode='r')
                tif = tifffile.memmap(directory/(name+'.tiff'), shape=original.shape, dtype=original.dtype,
                                      bigtiff=original.nbytes > 2**32-2**25,
                                      photometric='minisblack',
                                      metadata={'axes': 'YX', 'plane': p['id'], 'spacing_mm': p['spacing_mm'],
                                                'origin_plane_mm': p['origin_plane_mm']})
                for y in range(0, original.shape[0], 256):
                    tif[y:y+256] = original[y:y+256]
                tif.flush()
                del tif, original
        shutil.copyfile(result/'report.json', work/'report.json')
        work.rename(target)
        return {'status': report['status'], 'directory': str(target), 'format': 'TIFF + report.json'}
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
