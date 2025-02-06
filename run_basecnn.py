import torch

from src.model.cnn import BaseCnn
from src.model.inference import Inferencer
from src.model.training import CNNTraining
from src.datasets.data import SynDataset
from options import args

if __name__ == '__main__':
    

    args_dict = {
        'lr': args.lr,
        'conv1': args.conv1,
        'conv2': args.conv2,
        'pool1': args.pool1,
        'pool2': args.pool2,
        'fc2': args.fc2,
        'batch_size': args.batchsize,
        'dropout': args.dropout,
    }

    
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
    
    if args.pretrained:
        
        params = Inferencer.load_config(args.pretrained)
        cnn = BaseCnn.init_from_dict(params, args.pretrained)
        cnn.device = torch.device(args.device)
        cnn.to(cnn.device)
        cnn.batch_size = args.batchsize
    
    else:
 
        cnn = BaseCnn(dim_inp_x=args.input_size,
                    dim_inp_y=args.input_size,
                    n_channels=args.n_channels,
                    conv1_dict=args.conv1,
                    conv2_dict=args.conv2,
                    pool1=args.pool1,
                    pool2=args.pool2,
                    fc2=args.fc2,
                    dropout=args.dropout,
                    batch_size=args.batchsize,
                    device=args.device,
                    pretrained=args.pretrained)
        
    trainer = CNNTraining(model=cnn,
                            data=data,
                            optim_args=args.optim,
                            epochs=args.epochs,
                            model_name=args.model_name,
                            model_type=args.model_out,
                            log_result=args.log_result)


    if not args.pretrained:
    
        print(f'\nrunning training for: \n{trainer.model}\n')
        best_acc = trainer.train_model(args.save_opt)
    
    trainer.test_model()

print('done')