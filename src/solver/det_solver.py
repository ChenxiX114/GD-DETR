'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg
        self._print_hyperparams(args)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        best_ap = -1.0
        best_epoch = -1
        epochs_without_improvement = 0
        patience = getattr(args, 'patience', -1)
        for epoch in range(self.last_epoch + 1, args.epoches):
            # Update current_epoch for modules that need it (e.g. arch warmup, PAGD)
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'current_epoch'):
                self.model.encoder.current_epoch = epoch
            if hasattr(self.model, 'zoom') and self.model.zoom is not None:
                self.model.zoom.current_epoch.fill_(epoch)
            if hasattr(self.criterion, 'current_epoch'):
                self.criterion.current_epoch = epoch
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler,
                accum_steps=getattr(args, 'accum_steps', 1))

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # Update best stats
            for k in test_stats.keys():
                if k in best_stat:
                    if test_stats[k][0] > best_stat[k]:
                        best_stat[k] = test_stats[k][0]
                        best_stat['epoch'] = epoch
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            # Save best model weights when new best AP is achieved
            current_ap = test_stats.get('coco_eval_bbox', [0])[0]
            if current_ap > best_ap:
                best_ap = current_ap
                best_epoch = epoch
                epochs_without_improvement = 0
                if self.output_dir and dist.is_main_process():
                    dist.save_on_master(self.state_dict(epoch), self.output_dir / 'best.pth')
                    print(f'New best AP: {best_ap:.4f} @ epoch {epoch}, saved best.pth')
            else:
                epochs_without_improvement += 1
                if patience > 0:
                    print(f'No improvement for {epochs_without_improvement}/{patience} epochs')

            # Early stopping
            if patience > 0 and epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch}: no improvement for {patience} epochs. Best AP: {best_ap:.4f}')
                break

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            if hasattr(self.model, 'zoom') and self.model.zoom is not None:
                log_stats['zoom_gate'] = self.model.zoom.applied_gate()

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # Save last epoch checkpoint
        if self.output_dir:
            dist.save_on_master(self.state_dict(args.epoches - 1), self.output_dir / 'last.pth')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # Print and save best result summary
        print(f'\nBest result: AP={best_ap:.4f} @ epoch {best_epoch}')
        if self.output_dir and dist.is_main_process():
            best_summary = {
                'best_ap': round(best_ap, 4),
                'best_epoch': best_epoch,
                'total_epochs': epoch + 1,
                'total_time': total_time_str
            }
            # Append best stats from best_stat dict
            for k, v in best_stat.items():
                if isinstance(v, (int, float)):
                    best_summary[f'best_{k}'] = round(v, 4)
            with (self.output_dir / "log.txt").open("a") as f:
                f.write("\n" + json.dumps(best_summary) + "\n")


    def _print_hyperparams(self, args):
        """Print and save all training hyperparameters at start."""
        import pprint
        import io

        lines = []
        lines.append("=" * 60)
        lines.append("HYPERPARAMETERS")
        lines.append("=" * 60)

        params = {}
        for attr in sorted(dir(args)):
            if attr.startswith('_') or attr == 'yaml_cfg':
                continue
            try:
                val = getattr(args, attr)
                if not callable(val) and val is not None:
                    params[attr] = val
            except:
                pass

        buf = io.StringIO()
        pprint.pprint(params, width=80, compact=False, stream=buf)
        lines.append(buf.getvalue().rstrip())

        # Train dataloader info
        lines.append(f"\n--- DataLoader ---")
        lines.append(f"  train batch_size : {self.train_dataloader.batch_size}")
        lines.append(f"  train num_workers: {self.train_dataloader.num_workers}")
        lines.append(f"  val   batch_size : {self.val_dataloader.batch_size}")
        lines.append(f"  val   num_workers: {self.val_dataloader.num_workers}")

        # Optimizer info
        lines.append(f"\n--- Optimizer ---")
        for i, pg in enumerate(self.optimizer.param_groups):
            lines.append(f"  group {i}: lr={pg['lr']:.2e}, weight_decay={pg.get('weight_decay', 0)}")

        # Scheduler info
        if self.lr_scheduler:
            lines.append(f"\n--- LR Scheduler ---")
            lines.append(f"  type: {type(self.lr_scheduler).__name__}")
            lines.append(f"  initial lr: {self.lr_scheduler.get_last_lr()}")

        # Model param count
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        lines.append(f"\n--- Model ---")
        lines.append(f"  trainable params: {n_parameters:,}")
        lines.append(f"  total params:     {total_params:,}")

        # FLOPs calculation (PyTorch native, fast and accurate)
        try:
            from torch.utils.flop_counter import FlopCounterMode
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                dummy = torch.randn(1, 3, 640, 640, device=device)
            else:
                dummy = torch.randn(1, 3, 640, 640)
            self.model.eval()
            fc = FlopCounterMode(display=False)
            with fc:
                self.model(dummy)
            flops = fc.get_total_flops()
            self.model.train()
            if flops >= 1e9:
                lines.append(f"  FLOPs:            {flops/1e9:.2f} G")
            else:
                lines.append(f"  FLOPs:            {flops/1e6:.2f} M")
        except Exception as e:
            lines.append(f"  FLOPs:            N/A")

        lines.append("=" * 60)

        # Print to console
        content = "\n".join(lines)
        print("\n" + content + "\n")

        # Save to file
        if self.output_dir and dist.is_main_process():
            with open(self.output_dir / "parameters.txt", "w") as f:
                f.write(content + "\n")
            print(f"Saved parameters to {self.output_dir / 'parameters.txt'}")

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
