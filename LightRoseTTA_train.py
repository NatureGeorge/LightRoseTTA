'''
Protein Structure Prediction Network Train File
'''

import os
import time
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from confs import util
from utils.loss_func import harmonic_bond_force_loss
from utils.loss_func import harmonic_angle_force_loss
from utils.loss_func import periodic_torsion_force_loss
from utils.rigid_transform import transform_pred_coor_to_label_coor
from model.LightRoseTTA import Predict_Network
from utils.LDDT_torch import lddt as calculate_lddt
from data_pipeline import Protein_Dataset
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def cross_loss_mask(pred_, true_, mask, device):
    '''pair distance and angles cross entropy loss

    Input:
        - pred_(tensor):predict distance, omega, theta and phi
        - true_(tensor):true distance, omega, theta and phi
        - mask(tensor):distance, omega, theta and phi mask
        - device(str):cpu or cuda
    Output:
        - result(tensor):loss value
    '''
    pred_ = pred_.reshape(-1, pred_.shape[-1])
    true_ = torch.flatten(true_).to(device)
    mask = torch.flatten(mask).float().to(device)
    cross_func = torch.nn.CrossEntropyLoss(reduction='none')

    # import pdb
    # pdb.set_trace()
    loss = cross_func(pred_, true_)
    loss = mask * loss
    result = torch.mean(loss)
    return result

def coor_loss_func(pred_coor, target_coor):
    # coor loss function
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    return loss_fn(pred_coor, target_coor)

def templ_loss_func(msa_templ, real_templ):
    # coor loss function
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    return loss_fn(msa_templ, real_templ)

def lddt_loss_func(pred_lddt, true_lddt):
    
    lddt_mse_loss = torch.nn.MSELoss(reduction='none')
    return lddt_mse_loss(pred_lddt, true_lddt)

#dataset split
def data_builder(args, data_path, test_mode):

    '''build data
    Input:
        - args(object):arguments
        - data_path(str):train data path
        - test_mode(str):test or train mode
    Output:
        - train_loader(DataLoader):train data loader
        - val_loader(DataLoader):validation data loader
    '''
    dataset = Protein_Dataset(data_path, test_mode)

    args.num_classes = dataset.my_num_classes
    args.num_features = dataset.my_num_features
    args.num_bonds   = 0 # type of chemical bonds
    

    num_training = int(len(dataset)*0.95)
    num_val = int(len(dataset)*0.05)

    training_set,validation_set = random_split(dataset,[num_training,num_val])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size,shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader





def protein_loss_function(model, batchs, mode, args, epoch=0):
    '''compute all loss

    Input:
        - model:train model 
        - batchs:train data batch
        - loss_fn:mse loss function
        - lddt_mse_loss:lddt loss function
        - mode:train or test mode
        - args:arguments
    Output:
        - loss(tensor):sum loss value
    '''
    batchs = batchs.to(args.device)

    
    print('-----protein name-----:',batchs.protein_name)
    print("seq length:", batchs.msa.size()[-1])
    if mode == 'train':
        test_flag = False
    else:
        test_flag = True
    xyz, lddt, logits = model(batchs, test_flag)

    dis_pred, omega_pred, theta_pred, phi_pred = logits
    dis_label = torch.from_numpy(batchs.pair_prob_s_label[0][0])
    omega_label = torch.from_numpy(batchs.pair_prob_s_label[1][0])
    theta_label = torch.from_numpy(batchs.pair_prob_s_label[2][0])
    phi_label = torch.from_numpy(batchs.pair_prob_s_label[3][0])

    dis_mask = torch.from_numpy(batchs.pair_masks[0])

    dis_loss = cross_loss_mask(dis_pred.float(), dis_label.long(), dis_mask, args.device)
    omega_loss = cross_loss_mask(omega_pred.float(), omega_label.long(), dis_mask, args.device)
    theta_loss = cross_loss_mask(theta_pred.float(), theta_label.long(), dis_mask, args.device)
    phi_loss = cross_loss_mask(phi_pred.float(), phi_label.long(), dis_mask, args.device)

    n_target_coor  = torch.index_select(batchs.pos, dim=0, index=torch.subtract(batchs.CA_atom_index, 1))
    ca_target_coor = torch.index_select(batchs.pos, dim=0, index=batchs.CA_atom_index)
    o_target_coor  = torch.index_select(batchs.pos, dim=0, index=torch.add(batchs.CA_atom_index, 1))

    bb_pos = torch.stack([n_target_coor, ca_target_coor, o_target_coor], dim=1).reshape(-1, 3)
    new_out_coor = transform_pred_coor_to_label_coor(xyz, bb_pos, args)
    loss_coor = torch.sqrt(coor_loss_func(new_out_coor, bb_pos)+1e-8)

    e_bond_force_loss = harmonic_bond_force_loss(new_out_coor, batchs.bond_value, batchs.bond_index.T)

    e_angle_force_loss = harmonic_angle_force_loss(new_out_coor, batchs.angle_value, batchs.angle_index)

    e_dihedral_force_loss = periodic_torsion_force_loss(new_out_coor, bb_pos, batchs.dihedral_index)


    

    ca_out_coor = new_out_coor[1:new_out_coor.shape[0]-1:3]
    ca_target_coor = torch.index_select(batchs.pos, dim=0, index=batchs.CA_atom_index)

    LDDT_value = calculate_lddt(ca_out_coor.unsqueeze(0), ca_target_coor.unsqueeze(0), args, 15.0, True)
    LDDT_value = LDDT_value.to(args.device)
    loss_LDDT = torch.sqrt(lddt_loss_func(LDDT_value.squeeze(), lddt.squeeze().float())+1e-8)
    loss_LDDT = torch.mean(loss_LDDT)

    import pdb
    pdb.set_trace()

    if mode == 'train':
        print('''train coor loss:%.2f,  LDDT loss: %.2f, dis loss: %.2f, 
                omega loss: %.2f, theta loss: %.2f, phi loss: %.2f, 
                bond force loss: %.2f, angle force loss: %.2f, 
                dihedral force loss: %.2f'''
                %(loss_coor.item(), loss_LDDT.item(), dis_loss.item(), omega_loss.item(), 
                theta_loss.item(), phi_loss.item(), e_bond_force_loss.item(), e_angle_force_loss.item(), 
                e_dihedral_force_loss.item()))
    elif mode == 'test':
        print('''vaild coor loss:%.2f,  LDDT loss: %.2f, dis loss: %.2f, 
                omega loss: %.2f, theta loss: %.2f, phi loss: %.2f, 
                bond force loss: %.2f, angle force loss: %.2f, 
                dihedral force loss: %.2f'''
                %(loss_coor.item(), loss_LDDT.item(), dis_loss.item(), omega_loss.item(), 
                    theta_loss.item(), phi_loss.item(), e_bond_force_loss.item(), e_angle_force_loss.item(),
                    e_dihedral_force_loss.item()))


    loss = sum([0.5*loss_coor, 0.01*loss_LDDT, 0.3*dis_loss, 
                0.5*omega_loss, 0.5*theta_loss, 0.5*phi_loss,         
                (epoch+1)*0.005*e_bond_force_loss, 
    (epoch+1)*0.005*e_angle_force_loss
            , (epoch+1)*0.005*e_dihedral_force_loss])
                


    return loss, loss_coor


def test(model, loader, args, epoch_inside, train_max_seq_len=270):
    '''validation function

    Input:
        - model:network model
        - loader:valid data loader
        - args:arguments
        - loss_fn:msa loss function
        - lddt_mse_loss:LDDT loss function
    Output:
        - loss / (len(loader.dataset) - wrong_num):test loss
    '''
    model.eval()
    mode = 'test'
    loss = 0.
    wrong_num = 0
    coor_loss_sum = 0.
    for i, batchs in enumerate(tqdm(loader)):
        torch.cuda.empty_cache()
        if batchs.msa.shape[-1] > train_max_seq_len:
            continue
        
        with torch.no_grad():

            all_loss, coor_loss = protein_loss_function(model, batchs, mode, args, epoch_inside)
            loss += all_loss.item()
            coor_loss_sum += coor_loss.item()
     

        
        
    return loss / (len(loader.dataset) - wrong_num), coor_loss_sum / (len(loader.dataset) - wrong_num)

def main():
    '''
    train main function
    '''
    #parameter initialization
    parser = util.parser
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    #device selection
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    #test_mode_flag
    test_mode = False

    #training configuration
    train_loader, val_loader = data_builder(args, args.dataset, test_mode)
    
    model = Predict_Network(args).to(args.device)


    model_loss_file = "model_valid_loss.txt"

    accumulation_steps = 4

    train_max_seq_len = 270

    offset = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, 1, gamma=0.5)    

    with open(model_loss_file, 'a') as f:
        f.write("train begin")
        f.write('\n')

    #training steps
    patience = 0
    for epoch in range(args.epochs):
        
        model.train()
        mode = 'train'
        if (epoch+offset) > 7:
            epoch_inside = 7
        else:
            epoch_inside = epoch+offset
        for i, batchs in enumerate(tqdm(train_loader)):
            torch.cuda.empty_cache()
            print('-----protein name-----',batchs.protein_name[0])
            print('-----seq len-----',batchs.msa.shape[-1])
            if batchs.msa.shape[-1] > train_max_seq_len:
                print('overlong seq')
                continue

            sum_loss, _ = protein_loss_function(model, batchs, mode, args, epoch_inside)


            print("Training loss:{}".format(sum_loss.item()))
            print("Epoch{}".format(epoch))
            sum_loss.requires_grad_(True)
            # 2.1 loss regularization
            sum_loss = sum_loss / accumulation_steps
            # back propagation
            sum_loss.backward()

            if ((i+1)%accumulation_steps)==0:
                # optimizer the net
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                print("------       warning update the model       -------")
                optimizer.step()
                optimizer.zero_grad()


        torch.save(model.state_dict(),'weights/LightRoseTTA_{}.pth'.format(epoch+offset))
        val_loss, val_coor_loss = test(model, val_loader, args, epoch_inside, train_max_seq_len)
        print("----------------Validation loss:{}".format(val_loss))
        print("Epoch{}".format(epoch))
    
        print("--------------------------------------------Model saved at epoch{}".format(epoch))
        with open(model_loss_file, 'a') as f:
            f.write(time.strftime('%Y-%m-%D %H-%M-%S',time.localtime(time.time())))
            f.write("epoch {},loss {}, coor loss {}".format(epoch+offset, val_loss, val_coor_loss))
            f.write('\n')
        if val_loss < args.min_loss:
            args.min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > 3:
            scheduler.step()
            patience = 0



if __name__ == "__main__":
    main()
