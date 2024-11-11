import torch
from custom_datasets import OnlineFontSquare, HFDataCollector, TextSampler, FixedTextSampler, dataset_factory, GibberishSampler
from custom_datasets.real_datasets import transforms as T
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from utils import MetricCollector
from torchvision.utils import make_grid, save_image
from emuru import Emuru
import pickle
import wandb

def crop_width(width):
    def _inner(sample):
        img, txt = sample
        img = img[..., :width]
        return img, txt
    return _inner

def train(args):
    if args.device == 'cpu':
        print('WARNING: Using CPU')

    model = Emuru(args.t5_checkpoint, args.vae_checkpoint, args.ocr_checkpoint, args.slices_per_query).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.resume:
        try:
            checkpoint_path = sorted(Path(args.resume_dir).rglob('*.pth'))[-1]
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1 if args.resume_wandb else args.start_epoch
            args.wandb_id = checkpoint['wandb_id'] if args.resume_wandb else args.wandb_id
            print(f'Resumed training from {checkpoint_path}')
        except KeyError:
            model.load_pretrained(args.resume_dir)
            print(f'Resumed with the old checkpoint system: {checkpoint_path}')

    transform = T.Compose([
        # T.ToPILImage(),
        # T.PadMinWidth(max(kwargs['style_patch_width'], kwargs['dis_patch_width']), padding_value=255),
        # T.RandomShrink(1.0, 1.0, min_width=max(kwargs['style_patch_width'], kwargs['dis_patch_width']), max_width=gen_max_width, snap_to=gen_patch_width),
        T.ToTensor(),
        # T.MedianRemove(),
        T.Normalize((0.5,), (0.5,)),
        crop_width(768)

    ])

    dataset = dataset_factory('train', args.train_datasets, root_path=args.root_dir, post_transform=transform)
    dataset.batch_keys('style')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn,
                        num_workers=args.dataloader_num_workers)
    
    eval_dataset = dataset_factory('test', ['iam_lines'], root_path=args.root_dir)
    eval_dataset.batch_keys('style')
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn,
                            num_workers=args.dataloader_num_workers_eval, persistent_workers=False)

    # eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
    # dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))
    # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, collate_fn=model.data_collator, num_workers=args.dataloader_num_workers)

    if args.wandb:
        import wandb
        args.wandb_id = wandb.util.generate_id() if not hasattr(args, 'wandb_id') else args.wandb_id
        # resume = 'must' if args.resume_wandb else 'allow'
        resume = 'allow'
        wandb.init(project='Emuru', name=Path(args.output_dir).name, config=args, id=args.wandb_id, resume=resume)
    collector = MetricCollector()

    loader_iter = iter(loader)
    model.alpha = args.start_alpha
    for epoch in range(args.start_epoch, args.num_train_epochs):
        model.train()
        
        for i in tqdm(range(args.dataloader_chunk), desc=f'Epoch {epoch}'):
            try:
                # Get the next batch
                batch = next(loader_iter)
            except StopIteration:
                # If the iterator is exhausted, reinitialize it
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            res = model.tokenizer(batch['style_text'], return_tensors='pt', padding=True, return_attention_mask=True, return_length=True)
            res = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in res.items()}
            batch['noise'] = args.teacher_noise

            losses, pred, gt = model(img=batch['style_img'], **res)

            losses['loss'].backward()
            if (epoch * args.dataloader_chunk + i) % args.gradient_acc == 0:
                optimizer.step()
                optimizer.zero_grad()

            losses = {f'train/{k}': v for k, v in losses.items()}
            collector.update(losses)

            # print('Warning')
            # if i > 2:
            #     break
            # imgs = model.custom_generate(text='this is a sample text', img=None, max_new_tokens=96)
            # imgs = model.custom_generate(input_ids=batch['input_ids'], img=batch['img'], max_new_tokens=96 - 16, decoder_truncate=16)
            # print()

        with torch.no_grad():
            model.eval()
            wandb_data = {}
            wandb_data['train/alpha'] = model.alpha

            batch['input_ids'] = model.tokenizer(batch['style_text'], return_tensors='pt', padding=True).input_ids.to(args.device)
            batch['img'] = batch['style_img']
            if args.wandb:
                pred, gt, synth_gen_test = model.continue_gen_test(gt, batch, pred)
                # alpha = torch.ones_like(batch['img'][:, :1])
                # img_rgba = torch.cat([batch['img'], alpha], dim=1)
                gt = gt.repeat(1, 3, 1, 1)
                pred = pred.repeat(1, 3, 1, 1)
                synth_img = torch.cat([batch['style_img'], gt, pred], dim=-1)[:16]
                wandb_data['synth_img'] = wandb.Image(make_grid(synth_img, nrow=1, normalize=True))
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
            if args.wandb:
                pred, gt, real_gen_test = model.continue_gen_test(gt, batch, pred)
                # alpha = torch.ones_like(batch['img'][:, :1])
                # img_rgba = torch.cat([batch['img'], alpha], dim=1)
                gt = gt.repeat(1, 3, 1, 1)
                pred = pred.repeat(1, 3, 1, 1)
                real_img = torch.cat([batch['img'], gt, pred], dim=-1)[:16]
                wandb_data['real_img'] = wandb.Image(make_grid(real_img, nrow=1, normalize=True))
                wandb_data['real_gen_test'] = wandb.Image(real_gen_test)
                wandb.log(wandb_data | collector.dict())
                
        if epoch % 5 == 0 and epoch > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'wandb_id': args.wandb_id if args.wandb else None
            }
            checkpoint_path = Path(args.output_dir) / f'{epoch // 100 * 100:05d}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f'Saved model at epoch {epoch} in {checkpoint_path}')

        collector.reset()
        model.alpha -= args.decrement_alpha
        model.alpha = max(args.end_alpha, model.alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a T5 model with a VAE')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--t5_checkpoint', type=str, default='google-t5/t5-small', help='T5 checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default='results_vae/a912/model_0205', help='VAE checkpoint')
    parser.add_argument('--ocr_checkpoint', type=str, default='files/checkpoints/Origami_bw_img/origami.pth', help='OCR checkpoint')
    parser.add_argument('--resume_dir', type=str, default=None, help='Resume directory')
    parser.add_argument('--output_dir', type=str, default='files/checkpoints/Emuru_100k', help='Output directory')
    parser.add_argument('--root_dir', type=str, default='/home/vpippi/Teddy/files/datasets/', help='Output directory')
    parser.add_argument('--fonts', type=str, default='files/font_square/clean_fonts', help='Fonts path')
    parser.add_argument('--backgrounds', type=str, default='files/font_square/backgrounds', help='Backgrounds path')
    parser.add_argument('--renderers', type=str, help='Renderers path')
    parser.add_argument('--checkpoint_tag', type=str, default='', help='Checkpoint tag')
    parser.add_argument('--db_multiplier', type=int, default=1, help='Dataset multiplier')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=10 ** 10, help='Number of train epochs')
    parser.add_argument('--report_to', type=str, default='none', help='Report to')
    parser.add_argument('--dataloader_chunk', type=int, default=2000, help='Dataloader chunk size')
    parser.add_argument('--dataloader_num_workers', type=int, default=15, help='Dataloader num workers')
    parser.add_argument('--dataloader_num_workers_eval', type=int, default=4, help='Dataloader num workers')
    parser.add_argument('--slices_per_query', type=int, default=1, help='Number of slices to predict in each query')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id(), help='Wandb id')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--teacher_noise', type=float, default=0.1, help='How much noise add during training')
    parser.add_argument('--start_alpha', type=float, default=1.0, help='Alpha between the mse_loss (alpha=1) and the ocr_loss (alpha=0)')
    parser.add_argument('--end_alpha', type=float, default=1.0, help='Variable alpha')
    parser.add_argument('--decrement_alpha', type=float, default=0., help='Variable alpha')
    parser.add_argument('--gradient_acc', type=int, default=1)
    parser.add_argument('--train_datasets', type=str, nargs='+', default=['iam_lines', 'iam_words', 'iam_lines_xl'])
    args = parser.parse_args()

    if args.resume_dir is None:
        args.resume_dir = args.output_dir
    args.resume_wandb = args.resume_dir == args.output_dir

    train(args)