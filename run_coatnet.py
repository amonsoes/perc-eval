from options import args
from src.model.coatnet import CoAtNet
from src.model.training import CNNTraining
from src.datasets.data import SynDataset


if __name__ == '__main__':

    if args.dataset == '140k_flickr_faces':
        data = SynDataset(dataset=args.dataset,
                    device=args.device,
                    batch_size=args.batchsize,
                    transform=args.transform,
                    input_size=224,
                    adversarial_opt=args.adversarial_opt,
                    greyscale_opt=args.greyscale_opt)   
    elif args.dataset == 'debug':
        data = SynDataset(dataset=args.dataset,
                    device=args.device,
                    batch_size=args.batchsize,
                    transform=args.transform,
                    input_size=224,
                    adversarial_opt=args.adversarial_opt,
                    greyscale_opt=args.greyscale_opt)  
    else:
        raise ValueError('DATASET NOT SUPPORTED')

    coatnet = CoAtNet(image_size=(args.input_size,args.input_size),
                    pretrained=args.pretrained,
                    in_channels=args.n_channels,
                    channels=args.num_channels,
                    num_blocks=args.num_blocks,
                    device=args.device,
                    num_classes=1)
    
    coatnet.device = args.device
    
    trainer = CNNTraining(model=coatnet,
                            data=data,
                            optim_args=args.optim,
                            epochs=args.epochs,
                            model_type='coatnet',
                            model_name='CoAtNet',
                            log_result=args.log_result)


    if not args.pretrained:
        print(f'\nrunning training for: \n{trainer.model}\n')
        best_acc = trainer.train_model(args.save_opt)
    
    trainer.test_model()

print('done')