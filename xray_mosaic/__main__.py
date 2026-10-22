"""python -m xray_mosaic {inspect,run,calibrate,refine} ..."""
import argparse
import json
from pathlib import Path
from .geometry import GeometryError


def main():
    parser = argparse.ArgumentParser(description='Offline reference-plane X-ray mosaic')
    parser.add_argument('operation', choices=['inspect', 'run', 'calibrate', 'refine', 'export'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.operation == 'export':
            from .export import export_tiff
            if args.output is None:
                parser.error('export requires --output NEW_DIRECTORY')
            result = export_tiff(args.input, args.output)
        elif args.operation in ('run', 'inspect'):
            from .pipeline import Batch
            batch = Batch(args.input)
            if args.operation == 'run':
                if args.output is None:
                    parser.error('run requires --output NEW_DIRECTORY')
                result = batch.run(args.output)
            else:
                result = batch.inspect()
        else:
            data = json.loads(args.input.read_text(encoding='utf-8'))
            if args.operation == 'calibrate':
                from .calibration import fit_detector_sampling
                result = fit_detector_sampling(**data)
            else:
                from .registration import refine_plane_offsets
                result = refine_plane_offsets(**data)
            if args.output:
                with args.output.open('x', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (GeometryError, OSError, ValueError, TypeError) as e:
        print(json.dumps({'status': 'FAILED', 'diagnostic': str(e)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
