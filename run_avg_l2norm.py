
from tqdm import tqdm
from src.datasets.data import SynDataset
from options import args


if __name__ == '__main__':
    
    transform = 'calc_avg_attack_norm'

    if args.dataset == '140k_flickr_faces':
        data = SynDataset(dataset=args.dataset,
                    device=args.device,
                    batch_size=args.batchsize,
                    transform=transform,
                    input_size=224,
                    adversarial_opt=args.adversarial_opt,
                    greyscale_opt=args.greyscale_opt)   
    elif args.dataset == 'debug':
        data = SynDataset(dataset=args.dataset,
                    device=args.device,
                    batch_size=args.batchsize,
                    transform=transform,
                    input_size=224,
                    adversarial_opt=args.adversarial_opt,
                    greyscale_opt=args.greyscale_opt)  
    else:
        raise ValueError('DATASET NOT SUPPORTED')

    print(f'calculating L2 norm for:\n\nattack : {args.adversarial_opt.spatial_adv_type}\nepsilon : {args.eps}\ndata : {args.dataset} \n\n')
    iter = 0
    for x, y in tqdm(data.test):
        if iter == 317:
            break# calculation is done in Attack class. see adversarial/spatial.py for more
        iter += 1
    score = data.transforms.transform_val.adversarial_decider.transforms[0].transforms[1].get_avg_l2norm()
    
    print(f'\n\nAverage L2 norm for attack : {score} \n')
    
    
    print('done')