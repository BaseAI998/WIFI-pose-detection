"""Visual audit of human uncertain/standing labels; does NOT modify annotations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from ..data import resolve_recorded_path


def crop_tile(capture, sample, width=520, height=370):
    capture.set(cv2.CAP_PROP_POS_FRAMES,sample["source_frame"])
    ok,image = capture.read()
    if not ok:
        raise RuntimeError("Cannot decode label-review source frame")
    x1,y1,x2,y2 = sample["bbox_xyxy"]
    dx,dy = (x2-x1)*.3,(y2-y1)*.2
    x1,y1 = max(0,int(x1-dx)),max(0,int(y1-dy))
    x2,y2 = min(image.shape[1],int(x2+dx)),min(image.shape[0],int(y2+dy))
    crop = image[y1:y2,x1:x2]
    ratio = min(width/crop.shape[1],(height-55)/crop.shape[0])
    crop = cv2.resize(crop,(round(crop.shape[1]*ratio),round(crop.shape[0]*ratio)))
    canvas = np.full((height,width,3),30,np.uint8)
    ox,oy = (width-crop.shape[1])//2,55+(height-55-crop.shape[0])//2
    canvas[oy:oy+crop.shape[0],ox:ox+crop.shape[1]]=crop
    text = f"F{sample['output_frame']} P{sample['person_id']} {sample['source_time_sec']:.2f}s"
    cv2.putText(canvas,text,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(250,250,250),1,cv2.LINE_AA)
    text = f"HUMAN: {sample['label']} | {sample['occlusion']} | {sample['origin']}"
    cv2.putText(canvas,text,(8,45),cv2.FONT_HERSHEY_SIMPLEX,.48,(250,250,250),1,cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args = parser.parse_args()
    run = args.run_dir
    audit = json.loads((run/"data_audit.json").read_text(encoding="utf-8"))
    consistency = json.loads((run/"annotation_consistency.json").read_text(encoding="utf-8"))
    output = run/"label_review"
    output.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(resolve_recorded_path(
        audit["sources"]["video"]["path"], project_root=run.resolve().parents[1])))
    tiles = []
    try:
        for i,pair in enumerate(consistency["review_pairs"]):
            tile = np.hstack([crop_tile(capture,pair["uncertain"]),crop_tile(capture,pair["standing"])])
            cv2.imwrite(str(output/f"pair_{i:02d}.jpg"),tile)
            tiles.append(tile)
    finally:
        capture.release()
    # Paginate so labels are legible; no giant 10-pair image.
    for i in range(0,len(tiles),3):
        cv2.imwrite(str(output/f"consistency_page_{i//3+1}.jpg"),np.vstack(tiles[i:i+3]))
    print(f"Wrote {len(tiles)} candidate pairs to {output}; changed zero labels.")


if __name__ == "__main__":
    main()
