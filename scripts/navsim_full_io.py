"""Read every converted NAVSIM sample through METEOR's DataLoader.

Unlike BevLaneDataset.__getitem__, this check does not retry another frame
when a sample is unreadable. Run as a module/file (not via Python stdin) so
Windows multiprocessing workers can import the main module.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Subset

from bevlane.dataset import BevLaneDataset


class StrictBevLaneDataset(BevLaneDataset):
    def __getitem__(self, index):
        sample = self._get_one(index)
        if sample is None:
            scene, frame = self.items[index]
            raise RuntimeError(
                f"Unreadable NAVSIM sample: index={index}, scene={scene}, "
                f"frame={frame.get('frame')}, token={frame.get('token')}"
            )
        return sample


def init_worker(_worker_id):
    cv2.setNumThreads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only for a smoke test; 0 reads all frames")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--with-agenttraj", action="store_true")
    parser.add_argument("--with-command", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or args.batch_size < 1 or args.prefetch_factor < 1:
        parser.error("workers must be >= 0; batch-size and prefetch-factor must be >= 1")

    scenes = (args.root / "scenes.txt").read_text(encoding="utf-8").split()
    dataset = StrictBevLaneDataset(
        str(args.root), scenes, gt_key="gt",
        with_boxdet=not args.with_agenttraj,
        with_agenttraj=args.with_agenttraj,
        with_ego=True, with_depth=False, with_lidarbev=False,
        with_command=args.with_command,
    )
    expected = len(dataset)
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, expected)))
    target = len(dataset)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        prefetch_factor=args.prefetch_factor if args.workers else None,
        worker_init_fn=init_worker if args.workers else None,
    )
    print(json.dumps({"event": "start", "frames": target,
                      "total_dataset_frames": expected, "scenes": len(scenes),
                      "workers": args.workers, "batch_size": args.batch_size,
                      "prefetch_factor": args.prefetch_factor}),
          flush=True)
    start = time.perf_counter()
    count = 0
    next_progress = args.progress_every
    for batch in loader:
        images, intrinsics, extrinsics, gt = batch[:4]
        boxes, box_count = batch[4], batch[5]
        offset = 6
        trajectory = trajectory_valid = None
        if args.with_agenttraj:
            trajectory, trajectory_valid = batch[offset], batch[offset + 1]
            offset += 2
        ego = batch[offset]
        offset += 1
        command = batch[offset] if args.with_command else None
        offset += 1 if args.with_command else 0
        n = images.shape[0]
        if (images.shape[1:] != (8, 3, 432, 768)
                or intrinsics.shape[0] != n
                or extrinsics.shape[0] != n
                or gt.shape[0] != n
                or boxes.shape[0] != n
                or box_count.shape[0] != n
                or ego.shape[0] != n
                or offset != len(batch)
                or (trajectory is not None and trajectory.shape[1:] != (64, 6, 2))
                or (trajectory_valid is not None and trajectory_valid.shape[1:] != (64, 6))
                or (command is not None and command.shape[1:] != (3,))):
            raise RuntimeError(f"Bad batch shape at sample {count}")
        count += n
        if count >= next_progress or count == target:
            elapsed = time.perf_counter() - start
            print(json.dumps({"event": "progress", "frames": count,
                              "total": target, "elapsed_sec": round(elapsed, 2),
                              "frames_per_sec": round(count / elapsed, 3)}),
                  flush=True)
            next_progress = count + args.progress_every
    elapsed = time.perf_counter() - start
    if count != target:
        raise RuntimeError(f"Read {count} frames; expected {target}")
    print(json.dumps({"event": "complete", "frames": count,
                      "elapsed_sec": round(elapsed, 2),
                      "frames_per_sec": round(count / elapsed, 3)}), flush=True)


if __name__ == "__main__":
    main()
