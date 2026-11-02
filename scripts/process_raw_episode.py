"""Run one raw episode through trimming, segmentation and task-aware retiming."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_data_retime.automatic_dataset import remap_table
from real_robot_data_retime.timeline.smooth import sample_rows
from real_robot_data_retime.trim import analyze_episode, read_episode, source_episodes
from real_robot_data_retime.video import remap_video
from real_robot_data_retime.tasks import PROFILES


def dump(path, data):
    staged = path.with_suffix('.tmp.json')
    staged.write_text(json.dumps(data, indent=2) + '\n')
    staged.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def video_check(path, expected, fps):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    if count != expected or abs(actual_fps - fps) > 1e-4:
        raise ValueError(f'video clock mismatch: {path}: {count}/{expected}, {actual_fps}/{fps}')
    return dict(path=str(path), frames=count, fps=actual_fps, sha256=digest(path))


def inputs(args):
    source = args.source.resolve()
    info = json.loads((source / 'meta/info.json').read_text())
    row = next(row for row in source_episodes(source) if row['episode_index'] == args.episode)
    table = read_episode(source, info, row)
    for feature in ['action', 'observation.state']:
        values = np.asarray(table[feature].to_pylist())
        if values.shape != (len(table), 14) or not np.isfinite(values).all():
            raise ValueError(f'expected finite PiperX N x 14 {feature}')
    key = 'observation.images.right_environment_1' if args.task == 'water' else PROFILES[args.task]['camera']
    prefix = 'videos/' + key
    main = source / info['video_path'].format(video_key=key, chunk_index=row[prefix+'/chunk_index'], file_index=row[prefix+'/file_index'])
    identity = dict(task=args.task, episode=args.episode, source_repo=args.source_repo,
                    source_revision=args.source_revision, main_sha256=digest(main),
                    data_sha256=digest(source/info['data_path'].format(chunk_index=row['data/chunk_index'],file_index=row['data/file_index'])))
    return source, info, row, table, key, main, identity


def prepare(args):
    source, info, row, table, key, main, identity = inputs(args)
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('output must be outside the immutable source dataset')
    output.mkdir(parents=True, exist_ok=False)
    if args.task == 'water':
        from real_robot_data_retime.timeline.water import detect_phases
        phase = detect_phases(table['action'].to_pylist(), table['observation.state'].to_pylist(), info['fps'])
        dump(output/'detected-phases.json', phase)
        trim = dict(start=phase['head_removed'], stop=len(table), phase='task-aware water trimming')
    else:
        trim = analyze_episode(table['action'].to_pylist(), info['fps'])
    first = round(row['videos/'+key+'/from_timestamp'] * info['fps'])
    height, width = info['features'][key]['shape'][:2]
    remap_video(main, output/'trimmed.mp4', first+np.arange(trim['start'],trim['stop']), info['fps'], width, height)
    receipt = dict(identity=identity, trim=trim, source_frames=len(table), fps=info['fps'],
                   main_camera=key, trimmed=video_check(output/'trimmed.mp4',trim['stop']-trim['start'],info['fps']))
    dump(output/'prepared.json', receipt)
    print(json.dumps(receipt), flush=True)


def render(args):
    source, info, row, table, key, main, identity = inputs(args)
    output = args.output.resolve()
    prepared = json.loads((output/'prepared.json').read_text())
    if prepared['identity'] != identity:
        raise ValueError('source changed since preparation')
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    limit = args.gpu_memory_gib * 1024**3
    free, total = torch.cuda.mem_get_info(0)
    if free < limit + 512*1024**2:
        raise RuntimeError('insufficient unallocated GPU memory for the explicit shared-memory cap')
    torch.cuda.set_per_process_memory_fraction(limit/total, device=0)
    started = time.monotonic()
    resource = dict(gpu_memory_limit_gib=args.gpu_memory_gib, gpu_free_bytes_before=free,
                    gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader'],text=True).strip())
    dump(output/'execution.json', dict(status='running', identity=identity, resources=resource))
    if args.task == 'water':
        from real_robot_data_retime.water import main as water_main
        water_main(['render','--source',str(source),'--output',str(output/'water'),
                    '--episodes',str(args.episode),'--first-arm',args.water_first_arm,
                    '--delay-ratio',str(args.water_delay_ratio)])
        result = dict(output=str(output/'water/dataset'), task='water', validation=json.loads((output/'water/dataset/VALIDATION.json').read_text()))
    else:
        from real_robot_data_retime.edit import edit_video
        result = edit_video(output/'trimmed.mp4',output/'parallel.mp4',output/'analysis',args.task,point_tracker_device=args.point_tracker_device)
        if not result['success']:
            raise ValueError(f"interaction/render gates rejected this episode: {output/'analysis/report.json'}")
        with np.load(output/'analysis/source_mapping.npz') as mapping:
            left, right = mapping['left'], mapping['right']
        start, stop = prepared['trim']['start'],prepared['trim']['stop']
        trimmed = table.slice(start,stop-start)
        if any(np.any(np.diff(clock)<0) or clock.min()<0 or clock.max()>len(trimmed)-1 for clock in [left,right]):
            raise ValueError('non-monotone or out-of-source time map')
        fields=['action','observation.state','episode_index','frame_index','index','timestamp','task_index']
        exported = remap_table(trimmed.select(fields),left,right,0)
        exported = exported.set_column(exported.schema.get_field_index('timestamp'),'timestamp',pa.array(np.arange(len(left))/info['fps'],type=trimmed.schema.field('timestamp').type))
        pq.write_table(exported,output/'trajectories.parquet')
        for feature in ['action','observation.state']:
            raw = np.asarray(trimmed[feature].to_pylist())
            expected = np.c_[sample_rows(raw[:,:7],left),sample_rows(raw[:,7:],right)]
            actual = np.asarray(exported[feature].to_pylist())
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-5)
        videos=[video_check(output/'parallel.mp4',len(left),info['fps'])]
        for side,clock in [('left',left),('right',right)]:
            camera='observation.images.'+side+'_wrist';prefix='videos/'+camera
            original = source/info['video_path'].format(video_key=camera,chunk_index=row[prefix+'/chunk_index'],file_index=row[prefix+'/file_index'])
            first=round(row[prefix+'/from_timestamp']*info['fps'])
            height,width=info['features'][camera]['shape'][:2]
            path=output/(side+'_wrist.mp4')
            remap_video(original,path,first+start+clock,info['fps'],width,height)
            videos.append(video_check(path,len(left),info['fps']))
        result=dict(task=args.task,source_frames=len(table),trimmed_frames=len(trimmed),output_frames=len(left),
                    asynchronous_frames=int(np.sum(left!=right)),videos=videos,
                    trajectory_sha256=digest(output/'trajectories.parquet'),
                    validation='numeric alignment and all video frame counts passed; projected geometry only; visual review pending')
    result.update(elapsed_seconds=time.monotonic()-started,resources=resource,status='rendered',visual_review='pending')
    dump(output/'result.json',result)
    dump(output/'execution.json',dict(status='completed',identity=identity,result=result))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','render'])
    p.add_argument('--task',choices=['drawer','letters','workpiece','water'],required=True)
    for name in ['source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--source-repo',required=True)
    p.add_argument('--source-revision',required=True)
    p.add_argument('--gpu-memory-gib',type=float,default=3)
    p.add_argument('--point-tracker-device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--water-first-arm', choices=['left', 'right'], default='left')
    p.add_argument('--water-delay-ratio', type=float, default=0.5)
    a=p.parse_args()
    if a.gpu_memory_gib<=0:p.error('GPU limit must be positive')
    if not 0 <= a.water_delay_ratio <= 1:p.error('water delay ratio must be in [0,1]')
    (prepare if a.stage=='prepare' else render)(a)


if __name__=='__main__':main()
