from __future__ import annotations
import math
import sys
import traceback
from pathlib import Path
from .training import read_json, write_json


def run(config_path):
    config = read_json(Path(config_path))
    path = Path(config['path'])
    params = config['params']
    progress = {'epoch': 0, 'metrics': {}}
    try:
        from ultralytics import YOLO
        model = YOLO(params['model_path'])
        if model.task != 'detect':
            raise ValueError('仅支持 YOLO 目标检测模型')
        def epoch_end(trainer):
            metrics = {}
            for key, value in (trainer.metrics or {}).items():
                try:
                    number = float(value)
                    if math.isfinite(number):
                        metrics[key] = number
                except (ValueError, TypeError):
                    pass
            progress.update(epoch=int(trainer.epoch)+1, metrics=metrics)
            write_json(path/'progress.json', progress)
            print(f"完成轮次 {progress['epoch']}/{params['epochs']} · {metrics}", flush=True)
        model.add_callback('on_fit_epoch_end', epoch_end)
        options = {k: params[k] for k in ['epochs', 'imgsz', 'batch', 'device', 'lr0', 'patience', 'seed', 'hsv_s', 'hsv_v', 'fliplr', 'mosaic']}
        model.train(data=str(path/'dataset'/'dataset.yaml'), project=str(path/'output'), name='train', exist_ok=True, workers=0, optimizer='AdamW', amp=False, plots=False, save=True, **options)
        best = path/'output'/'train'/'weights'/'best.pt'
        if not best.is_file():
            raise RuntimeError('训练结束但未生成 best.pt')
        progress['model_path'] = str(best.resolve())
        write_json(path/'progress.json', progress)
    except BaseException as exc:
        progress['error'] = str(exc)
        write_json(path/'progress.json', progress)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run(sys.argv[1])
