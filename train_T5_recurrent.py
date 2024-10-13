import torch
from custom_datasets import OnlineFontSquare, TextSampler, dataset_factory
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
import argparse
from tqdm import tqdm
from utils import MetricCollector
from torchvision.utils import make_grid
from train_T5 import Emuru

def train(args):
    if args.device == 'cpu':
        print('WARNING: Using CPU')

    model = Emuru(args.t5_checkpoint, args.vae_checkpoint, args.ocr_checkpoint, args.slices_per_query).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.resume:
        try:
            checkpoint_path = sorted(Path(args.output_dir).rglob('*.pth'))[-1]
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            args.wandb_id = checkpoint['wandb_id']
            print(f'Resumed training from {args.output_dir}')
        except KeyError:
            model.load_pretrained(args.output_dir)
            print(f'Resumed with the old checkpoint system: {args.output_dir}')

    sampler = TextSampler(8, 32, (4, 7))
    # sampler = GibberishSampler(32)
    dataset = OnlineFontSquare('files/font_square/clean_fonts', 'files/font_square/backgrounds', sampler)
    dataset.length *= args.db_multiplier
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=model.data_collator,
                        num_workers=args.dataloader_num_workers, persistent_workers=args.dataloader_num_workers > 0)
    
    eval_dataset = dataset_factory('test', ['iam_lines'], root_path='/home/vpippi/Teddy/files/datasets/')
    eval_dataset.batch_keys('style')
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, collate_fn=eval_dataset.collate_fn,
                            num_workers=args.dataloader_num_workers_eval, persistent_workers=False)

    # eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
    # dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))
    # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, collate_fn=model.data_collator, num_workers=args.dataloader_num_workers)

    if args.wandb:
        wandb.init(project='Emuru', config=args, id=args.wandb_id)
    collector = MetricCollector()

    for epoch in range(args.start_epoch, args.num_train_epochs):
        model.train()
        
        # Create an iterator from the loader
        loader_iter = iter(loader)
        
        for i in tqdm(range(100), desc=f'Epoch {epoch}'):
            try:
                # Get the next batch
                batch = next(loader_iter)
            except StopIteration:
                # If the iterator is exhausted, reinitialize it
                loader_iter = iter(loader)
                batch = next(loader_iter)
            
            # Move tensors to the correct device
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['noise'] = args.teacher_noise

            losses, pred, gt = model.forward_recurrent(**batch)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()

            losses = {f'train/{k}': v for k, v in losses.items()}
            collector.update(losses)

        with torch.no_grad():
            model.eval()
            wandb_data = {}

            _, _, gt = model._img_encode(batch['img'])

            _, gt, synth_gen_test = model.continue_gen_test(gt, batch)
            wandb_data['synth_gen_test'] = wandb.Image(synth_gen_test)

            for i, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f'Eval'):
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                res = model.tokenizer(batch['style_text'], return_tensors='pt', padding=True, return_attention_mask=True, return_length=True)
                res = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in res.items()}

                losses, pred, gt = model(img=batch['style_img'], **res)
                losses = {f'eval/{k}': v for k, v in losses.items()}
                collector.update(losses)

            batch['input_ids'] = model.tokenizer(batch['style_text'], return_tensors='pt', padding=True).input_ids.to(args.device)
            batch['img'] = batch['style_img']
            pred, gt, real_gen_test = model.continue_gen_test(gt, batch, pred)
            real_img = torch.cat([batch['img'], gt, pred], dim=-1)[:16]
            wandb_data['real_img'] = wandb.Image(make_grid(real_img, nrow=1, normalize=True))
            wandb_data['real_gen_test'] = wandb.Image(real_gen_test)

            if args.wandb:
                wandb.log(wandb_data | collector.dict())
                
        if epoch % 5 == 0 and epoch > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'wandb_id': args.wandb_id
            }
            checkpoint_path = Path(args.output_dir) / f'{epoch // 100 * 100:05d}_recurr.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f'Saved model at epoch {epoch} in {checkpoint_path}')

        collector.reset()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a T5 model with a VAE')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--t5_checkpoint', type=str, default='google-t5/t5-small', help='T5 checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default='results_vae/abf3/model_0066', help='VAE checkpoint')
    parser.add_argument('--ocr_checkpoint', type=str, default='files/checkpoints/Origami_bw_img/origami.pth', help='OCR checkpoint')
    parser.add_argument('--output_dir', type=str, default='files/checkpoints/Emuru_100k', help='Output directory')
    parser.add_argument('--db_multiplier', type=int, default=1, help='Dataset multiplier')
    parser.add_argument('--learning_rate', type=float, default=0.000001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=10 ** 10, help='Number of train epochs')
    parser.add_argument('--report_to', type=str, default='none', help='Report to')
    parser.add_argument('--dataloader_num_workers', type=int, default=10, help='Dataloader num workers')
    parser.add_argument('--dataloader_num_workers_eval', type=int, default=4, help='Dataloader num workers')
    parser.add_argument('--slices_per_query', type=int, default=1, help='Number of slices to predict in each query')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id(), help='Wandb id')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--teacher_noise', type=float, default=0.1, help='Start epoch')
    args = parser.parse_args()

    train(args)