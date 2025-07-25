# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:18:00 2024

@author: P0112011
Code based on: https://github.com/Shef-AIRE/AI4Cardiothoracic-CardioVAE/blob/main/CardioVAE.ipynb

#3rd Paper!
To do:
    
1- Done 
[Note: the results in the abstract consider grouped by patient and all files. I need to investigate the cherry picking mentioned below]

2- More details:
    Main Results from new splits (seed 0 and 5-fold CV) <- Use it as a section on "Effect of Training Data"
    
3- And...?    
    Additional results for new splits (Sup. Materials?)
    Check other features?
    Plot results by lesion type? (everything will be on the same level I think)
    
Short notes about results:
    Grouping by file makes more sense than by Patients
    
    But then again, all analysis below are using the average of 254 lines per file, which gets tons of out-of-lesion data
    Benign x Malignant: 0.57 overall, 
                        0.55 for Iman's and 0.58 for Arnaud's selections
                        0.56 for trans  and 0.57 for sagittal views
                        0.60 for bigger, shallow lesions
    It still might be relevant to use Iman/Arnaud/trans/sag for the overall results (detection)
                        It doesn't change that much for detection using the old split...
    It will be interesting to get lesion-only data to recheck this analysis
                        We only really get differences (AUC > 0.6) for deeper lesions (max_diff, skewness and kurtosis)

    WE GOT HIGHER AUC FOR BIG LESIONS! AND LESS DEEP! (NEED CONFIRM AFTER BOOSTRAP/DELONG)
    No changes based on B-mode CNR
    
    The differences between QUS_in_NASH and Oncotech LESIONS is much higher than QUS_in_NASH and Oncotech LIVER
    AUC: 0.665, 95% CI: [0.558, 0.772] <- Using max_lesion (maximum diff within lesion) on Deeper Lesions
    
    There is a weak but significant correlation between lesion size and reconstruction error
        Pearson Correlation (0.189, p=0.003)
        Spearman Correlation (0.263, p=0.000)
        Maybe I should do lesion size but using the mask instead of the annotation
    
Previous comments:
model_load_checkpoint = True #I need to run seed 9 again with the checkpoint
To do:
        Investigate features/parameters other than reconstruction!
        Maybe check envelope signal -> it is less noisy! QUS_MIGR_1_ECHOENV.DAT


    See if we can improve the model in any way
- Write code that compares the “heat map” of the reconstruction error to the segmentation mask
Idea for that:
        We can get the start of liver coordinates (Arnaud) and remove from the reconstructed
        Then, we will know we are inside the liver (at least at the beginning)
        and we will remove the major noise/variation components
        
        Also
        use the segmentation masks to see the differences
        maybe try reconstruction/diff maps normalized or log-scaled.

"""




from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import loadmat
import random
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

import torch.nn as nn
from torch.autograd import Variable
# from scipy.signal import hilbert
# from skimage.transform import resize
from sklearn.metrics import roc_curve, roc_auc_score
from utils.results import replace_labels

import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)

# Define configuration
csv_file = "QUSinNASH_Dataset_20250217.csv"
# path_to_RF = "RF_dataset/RF_data/"
# path_to_masks = "RF_dataset/RF_masks/"
path_to_sets = "RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/"
saved_splits = os.listdir(path_to_sets)


data_mode = "RF" # only RF or RF-env for now
biomarker = "NAS_class"
net_mode = "load" # Options are "train" or "tune" or "load" or "TTA"
seed = 25 #25, 42, 21, 19, 9, 0
liveronly = False

# latent_size = 16 #important parameter for VAE according to https://papers.miccai.org/miccai-2024/paper/0616_paper.pdf
batch_size = 128

model_save = True
load_seeds = True

model_load_checkpoint = False #I need to run seed 9 again with the checkpoint
volunteers_only = False #True




#%% Split train/val for anomaly detection
def split_sets(csv_file, biomarker, test_size=0.25, random_state=21, volunteers=False):
    #Get the data from the csv file and remove empty values
    data = pd.read_csv(csv_file, delimiter=',')
    print('.csv file loaded. Number of RF files (total):', len(data))
    if volunteers:
        train = data.loc[data['Steatosis_percentage'] == "No biopsy"]
        val = data.loc[data['Steatosis_percentage'] != "No biopsy"]
        print('Using only volunteers for training: %i subjects' % len(pd.unique(train['Patient'])))
        print('Using patients for validation: %i subjects' % len(pd.unique(val['Patient'])))
    else:
        data = data.dropna(subset=[biomarker])
        patients_df = data.drop_duplicates(subset=['Patient'])
        train, val = train_test_split(patients_df, test_size=test_size, random_state=random_state)
        print('Number of files (included):', data.shape[0])
        print('Number of patients:', patients_df.shape[0])

    trainset = data[data['Patient'].isin(train['Patient'])]
    valset = data[data['Patient'].isin(val['Patient'])]
    print('Train set:', trainset.shape[0], 'files. | Unique patients:', trainset['Patient'].nunique())
    print('Validation set:', valset.shape[0], 'files. | Unique patients:', valset['Patient'].nunique())
    return trainset.reset_index(), valset.reset_index()

# Here to prepare and save sets, we can skip it if we have it trained
def acquire_mask(filename):
    # mask_file = filename[:-4] #Remove extension
    # mask_file = mask_file + '_EchoEnv.dat_mask2.dat'
    
    mask_file = filename
    
    fid = open(mask_file, 'r')
    size_Mask = np.fromfile(fid, np.int32,4)
    mask = np.fromfile(fid, np.int8,-1)
    mask = mask[:30*254*3249] #to avoid one patient with 100 frames
    mask2 = mask.reshape(30,254,3249).T
    return mask2

def load_RF(data, biomarker, data_mode, balance=False):
    all_files = data['Filepath']#os.listdir(path_to_RF)
    all_masks = data['Maskpath']
    images = []
    labels = []
    identifier = []
    print('Using mode:', data_mode)
    for i in range(len(data)):
        # Printing progress
        if i > 0 and i % 10 == 0:
            print('Loading file', i, 'of', len(data))
        filename = all_files[i] #data['Visit'][i] + '_' + data['Patient'][i]
        # matching_file = [file for file in all_files if filename in file][0]
        RF_data = loadmat(filename)['RFmig']
        mask = acquire_mask(all_masks[i]) #matching_file, path_to_mask)
        for j in range(0, 30, 5): #30 for all frames, or range(1) for just one frame
            single_RF = RF_data[...,j]
            norm_RF = (single_RF - np.mean(single_RF)) / np.std(single_RF)
            coords = np.where(mask[...,j] != 0)
            x1,y1,x2,y2 = np.min(coords[1]), np.min(coords[0]), np.max(coords[1]), np.max(coords[0])
            X_middle = int((x1 + x2)/2)

            # n_samples = 20 #a number or int(largest_x/2)
            # lines = np.arange(-n_samples, n_samples, 2)
            lines = np.arange(0, norm_RF.shape[1])
            for l in lines:
                # RF_line = norm_RF[:, X_middle+l]
                RF_line = norm_RF[:, l]
                # RF_line = RF_line - np.mean(RF_line) #remove mean?
                
                if 'env' in data_mode:
                    from scipy.signal import hilbert
                    envelope = np.abs(hilbert(RF_line))
                    RF_line = envelope
                    
                
                if liveronly: #crop region within liver, downsample to maintain size
                    line_mask = mask[:,X_middle+l,j]
    #                     masked_line = np.multiply(RF_line,1-line_mask) #for outside the liver
                    begin = np.where(line_mask>0)[0][0]
                    end = np.where(line_mask>0)[0][-1]
                    RF_line = RF_line[begin:end]
                    downsampling_factor = 4 #Same as Aiguo Han
                    ds_signal = RF_line[:-1:downsampling_factor]
                    images.append(np.resize(ds_signal, (400)))
                else:
                    images.append(RF_line[:-1])   #removing the last point so it's a multiple of 8                  
                    
                labels.append(data[biomarker][i])
                identifier.append(filename+'_%s_%s' % (j, l))
    X = np.array(images)
    y = np.array(labels)
    z = np.array(identifier)

    # if balance:
    #     print('Using balanced dataset')
    #     # Balance the dataset
    #     distribution = pd.Series(y).value_counts(normalize=False)
    #     print('Distribution of classes:', distribution.to_dict())
    #     least_common = distribution.idxmin()
    #     least_common_count = distribution.min()
    #     print('Least common class - %s: ' % biomarker, int(least_common), 'with', least_common_count, 'samples')

    #     # Select same number of samples for all classes
    #     balanced_X = []
    #     balanced_y = []
    #     for grade in np.unique(y):
    #         class_samples = X[y == grade]
    #         random_samples = random.sample(list(class_samples), least_common_count)
    #         balanced_X.extend(random_samples)
    #         balanced_y.extend([grade]*least_common_count)
    #     X = np.array(balanced_X)
    #     y = np.array(balanced_y)

    print('Data loaded with shape - X:', X.shape, 'y:', y.shape)
    return np.expand_dims(X, axis=1), y, z

if load_seeds == False:
    trainset, valset = split_sets(csv_file, biomarker, test_size=0.25, random_state=seed, volunteers=volunteers_only)
    print(trainset['NAS_class'].value_counts())
    print(valset['NAS_class'].value_counts())
    
    train = load_RF(trainset, biomarker, data_mode)
    val = load_RF(valset, biomarker, data_mode)
    print('Saving train and validation sets...')
    torch.save(train, path_to_sets + 'train_{}_seed-{}_Volunteers_fullsignal.pth'.format(data_mode, int(seed)))
    torch.save(val, path_to_sets + 'val_{}_seed-{}_Patients_fullsignal.pth'.format(data_mode, int(seed)))

#%% Loading is quicker when possible

# Choose 1 (Full Signal) or 2 (downsampled 400-points inside liver) or 3 (volunteers)
choice = 1

if load_seeds:
    if choice==1:
        print('Loading full signals...')
        train = torch.load(path_to_sets + 'train_{}_seed-{}_fullsignal_alllines.pth'.format(data_mode, int(seed)), weights_only=False)
        val = torch.load(path_to_sets + 'val_{}_seed-{}_fullsignal_alllines.pth'.format(data_mode, int(seed)), weights_only=False)
    elif choice==2:
        print('Loading downsampled signals...')
        train = torch.load(path_to_sets + 'train_{}_seed-{}_liveronly.pth'.format(data_mode, int(seed)), weights_only=False)
        val = torch.load(path_to_sets + 'val_{}_seed-{}_liveronly.pth'.format(data_mode, int(seed)), weights_only=False)
    elif choice==3:
        print('Loading signals split into volunteers and patients...')
        train = torch.load(path_to_sets + 'train_{}_seed-{}_Volunteers_fullsignal.pth'.format(data_mode, int(seed)), weights_only=False)
        val = torch.load(path_to_sets + 'val_{}_seed-{}_Patients_fullsignal.pth'.format(data_mode, int(seed)), weights_only=False)
    else:
        print('Instructions not clear?')
print('Shapes are:', train[0].shape, '--', val[0].shape)

#%% Looking at the files

sample_no = 15 #Patient number
frame_no = 2 #Frame number

### Given that we have 30 frames, and 20 lines per frame
# nframes = 30
# nlines = 20

### Given that we have x frames, and all lines
nframes = 6
nlines = 254 #254 or 50

sample_npoints = np.arange(sample_no*nframes*nlines,(sample_no+1)*nframes*nlines)
sample_patient = train[0][sample_npoints] # 600 signals in dim (1,3248) (if full signal)

frame_npoints = np.arange(frame_no*nlines, (frame_no+1)*nlines)
frame_RF = sample_patient[frame_npoints]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(5,4)
axes = axes.flatten()
for frame,ax in zip(frame_RF,axes):
    ax.plot(frame[0])
plt.suptitle('Each RF line')
plt.show()

RF_frame = np.transpose(np.squeeze(frame_RF, axis=1))

plt.imshow(RF_frame, cmap='gray', aspect='auto'), plt.title('RF'), plt.show()

from utils.loading_data import rf2bmode as r2b
Bmode_frame = r2b(RF_frame)

#To do: A better processing could improve the visualization -> check matlab scripts.
plt.imshow(Bmode_frame, cmap='gray', aspect='auto'), plt.title('Bmode'), plt.show()


#%% Data loaders
import matplotlib.pyplot as plt

plt.subplot(2,2,1)
plt.plot(train[0][0][0])
plt.title('Training - Grade %s' % train[1][0])
plt.subplot(2,2,2)
plt.plot(train[0][200][0])
plt.title('Training - Grade %s' % train[1][200])
plt.subplot(2,2,3)
plt.plot(val[0][-1][0])
plt.title('Validation - Grade %s' % val[1][-1])
plt.subplot(2,2,4)
plt.plot(val[0][0][0])
plt.title('Validation - Grade %s' % val[1][0])
plt.show()


def prepare_loaders(data, batch_size=32, shuffle=False, zeroesonly=False):
    X, y, z = data
    if zeroesonly:
        idx = np.where(y==0)[0]
        X, y, z = X[idx], y[idx], z[idx]
        
    X = X.astype(np.float32)
    # y = y.astype(np.int64)
    dataset = list(zip(X, y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, z

trainloader,_       = prepare_loaders(train, batch_size=batch_size, shuffle=True)
valloader,_         = prepare_loaders(val, batch_size=1, shuffle=False)
# valloader_zeroes,_  = prepare_loaders(val, batch_size=batch_size, shuffle=False, zeroesonly=True)
print('Done.')

#%% Model and Training functions
#Check CUDA availability
global device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', device)


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
    
class ECGVAEEncoder(nn.Module):
    def __init__(self, input_dim=60000, latent_dim=256):
        super(ECGVAEEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(in_features=64 * (input_dim // 8), out_features=latent_dim)  # Adjusted for stride=2, 3 layers
        self.fc_logvar = nn.Linear(in_features=64 * (input_dim // 8), out_features=latent_dim)
        # self.fc_mu = nn.Linear(in_features= 26048, out_features=latent_dim)  # Hard-coded value for signals of length = 3249
        # self.fc_logvar = nn.Linear(in_features= 26048, out_features=latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ECG Modality Decoder
class ECGVAEDecoder(nn.Module):
    def __init__(self, latent_dim=256, output_dim=60000):
        super(ECGVAEDecoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dim, out_features=64 * (output_dim // 8))
        self.convtrans1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.output_activation = nn.Identity()  # Suitable for standardized data

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, z.size(1) // 64)  # Adjust the reshape for proper dimensions
        z = self.relu(self.convtrans1(z))
        z = self.relu(self.convtrans2(z))
        z = self.output_activation(self.convtrans3(z))
        return z

class CardioVAE(nn.Module):
    def __init__(self, image_input_channels=1, ecg_input_dim=60000, latent_dim=256):
        super(CardioVAE, self).__init__()
        self.ecg_encoder = ECGVAEEncoder(ecg_input_dim, latent_dim)
        self.ecg_decoder = ECGVAEDecoder(latent_dim, ecg_input_dim)

        self.experts       = ProductOfExperts()
        self.n_latents     = latent_dim

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
          return mu

    def forward(self, image=None, ecg=None):
        mu, logvar = self.infer(image, ecg)
        # reparametrization trick to sample
        z          = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = 0#self.image_decoder(z)
        ecg_recon  = self.ecg_decoder(z)
        return img_recon, ecg_recon, mu, logvar

    def infer(self, image=None, ecg=None): 
        batch_size = image.size(0) if image is not None else ecg.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)
        if image is not None:
            img_mu, img_logvar = self.image_encoder(image)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if ecg is not None:
            ecg_mu, ecg_logvar = self.ecg_encoder(ecg)
            mu     = torch.cat((mu, ecg_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, ecg_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#set_seed(seed)

def elbo_loss(recon_xray, xray, recon_ecg, ecg, mu, logvar,
              lambda_xray=1.0, lambda_ecg=1.0, annealing_factor=1):

    xray_mse, ecg_mse = 0, 0
    if recon_xray is not None and xray is not None:
        # Reshape to the original image size
        xray_mse = nn.functional.mse_loss(recon_xray, xray, reduction='sum')

    if recon_ecg is not None and ecg is not None:
        # Reshape to the original image size
        ecg_mse = nn.functional.mse_loss(recon_ecg, ecg, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    ELBO = torch.mean(lambda_xray * xray_mse + lambda_ecg * ecg_mse + annealing_factor * KLD)
    
    return ELBO, ecg_mse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.list.append(self.avg)



#%% Train or load model
n_latents = 256
epochs = 100
annealing_epochs = 50
lr = 1e-3
# log_interval = 10
lambda_xray = 0.0
lambda_ecg = 10.0 # 10

patience = 200

if liveronly: data_mode += '-liveronly'
# biomarker = 'latent16_' + biomarker

validation = valloader #or valloader_zeroes


# Model and optimizer setup
model = CardioVAE(image_input_channels=1, ecg_input_dim=train[0].shape[-1],latent_dim=n_latents).to(device)
if model_load_checkpoint:
    checkpoint_file = 'trained_model_NAS_class_RF-liveronly_seed-42_final_UpTo3000.pth'
    print('We loaded the model with the weights from:', checkpoint_file)
    model.load_state_dict(torch.load(path_to_sets+checkpoint_file, weights_only=True))




if net_mode=='train':
    dir_savename = path_to_sets + 'trained_model_env_' + str(seed)
    os.makedirs(dir_savename, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Main training and validation loop
    best_loss = float('inf')
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    N_mini_batches = len(trainloader)
    train_list = []
    val_list = []
    n_since_best = 0
    
    for epoch in range(epochs):
        #Old annealing
        annealing_factor = min(epoch / annealing_epochs, 1) if epoch < annealing_epochs else 1.0
        #New annealing
        # annealing_factor = (epoch / epochs)

        
        #Training
        model.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            if inputs.shape[2]%2==1: inputs = inputs[:,:,:-1]
       
            optimizer.zero_grad()
            # recon_xray_joint, recon_ecg_joint, mu_joint, logvar_joint = model(xray, ecg)
            # recon_xray_only, _, mu_xray, logvar_xray = model(image=xray)
            _, recon_ecg_only, mu_ecg, logvar_ecg = model(ecg=inputs)
        
        
            # joint_loss = elbo_loss(recon_xray_joint, xray, recon_ecg_joint, ecg, mu_joint, logvar_joint, lambda_xray, lambda_ecg, annealing_factor)
            # xray_loss = elbo_loss(recon_xray_only, xray, None, None, mu_xray, logvar_xray, lambda_xray, lambda_ecg, annealing_factor)
            ecg_loss, mse_loss = elbo_loss(None, None, recon_ecg_only, inputs, mu_ecg, logvar_ecg, lambda_xray, lambda_ecg, annealing_factor)
        
        
            train_loss = ecg_loss #+joint_loss + xray_loss
            train_loss.backward()
            optimizer.step()
            
            train_loss_meter.update(train_loss.item(), len(inputs))
            # if batch_idx % log_interval == 0:
            #     print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(trainloader.dataset)} ({100. * batch_idx / N_mini_batches:.0f}%)]\tLoss: {train_loss_meter.avg*10e-6:.2f}')
            if batch_idx+1 == len(trainloader):
                print(f'Train Epoch: {epoch+1} \t\tLoss: {train_loss_meter.avg*10e-6:.2f}')
                
        train_list.append(train_loss_meter.avg)
    
        #Validation        
        model.eval()
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_labels) in enumerate(validation):
                val_inputs = val_inputs.to(device)
                if val_inputs.shape[2]%2==1: val_inputs = val_inputs[:,:,:-1]
                
                _, val_recon_ecg_only, val_mu_ecg, val_logvar_ecg = model(ecg=val_inputs)
                val_ecg_loss, val_mse = elbo_loss(None, None, val_recon_ecg_only, val_inputs, val_mu_ecg, val_logvar_ecg, lambda_xray, lambda_ecg, annealing_factor)
                val_loss = val_ecg_loss
                
                val_loss_meter.update(val_loss.item(), len(val_inputs))
                if val_batch_idx+1 == len(validation):
                    print(f'Val Epoch: {epoch+1} \t\tLoss: {val_loss_meter.avg*10e-6:.2f}')
                    print('ELBO: %.2f / MSE: %.2f / Annealing = %.2f' % (val_ecg_loss.item(), val_mse.item(), annealing_factor))

        val_list.append(val_loss_meter.avg)
        if val_loss_meter.avg < best_loss:
            best_loss = val_loss_meter.avg
            n_since_best = 0
            savemodel_filename = dir_savename+'/trained_model_{}_{}_seed-{}_bestloss.pth'.format(biomarker, data_mode, seed)
            torch.save(model.state_dict(),  savemodel_filename)
            print("Saved model current state dictionary to", savemodel_filename)
        
        if (epoch+1) % 1000 == 0:
            savemodel_checkpoint = dir_savename+'/trained_model_{}_{}_seed-{}_checkpoint_{}.pth'.format(biomarker, data_mode, seed, epoch)
            torch.save(model.state_dict(),  savemodel_checkpoint)
            print('Saved checkpoint')

        n_since_best += 1
        if n_since_best >= patience:
            break
             
    
    
    # After training is complete
    savemodel_filename = dir_savename+'/trained_model_{}_{}_seed-{}_final.pth'.format(biomarker, data_mode, seed)
    torch.save(model.state_dict(),  savemodel_filename)
    print("Saved final model state dictionary to", savemodel_filename)
    plt.plot(train_list), plt.plot(val_list), plt.ylim([np.min(train_list),np.max(train_list)]), plt.title(n_latents), plt.show()
    
if net_mode=='load':
    print('Loading weights...')
    weights_path = 'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed)
    model.load_state_dict(torch.load(path_to_sets+weights_path, weights_only=True))
    print('Done. Loaded: ', weights_path)

#%%# For plotting an example

if net_mode =='train':
    # input_a = np.array(inputs.cpu())[0][0]
    # recon_a = np.array(recon_ecg_only.cpu().detach().numpy())[0][0]
    # label_a = labels[0]
    # input_b = np.array(inputs.cpu())[81][0]
    # recon_b = np.array(recon_ecg_only.cpu().detach().numpy())[81][0]
    # label_b = labels[81]
    
    input_a = np.array(val_inputs.cpu())[0][0]
    recon_a = np.array(val_recon_ecg_only.cpu().detach().numpy())[0][0]
    label_a = val_labels[0].item()
    # input_b = np.array(val_inputs.cpu())[19][0]
    # recon_b = np.array(val_recon_ecg_only.cpu().detach().numpy())[19][0]
    # label_b = val_labels[19].item()
    
    
    plt.subplot(2,2,1)
    plt.plot(input_a)
    plt.title('Input A')
    # plt.subplot(2,2,2)
    # plt.plot(input_b)
    # plt.title('Input B')
    plt.subplot(2,2,3)
    plt.plot(recon_a)
    # plt.subplot(2,2,4)
    # plt.plot(recon_b)
    plt.show()
    print('Input A (mean, std):', np.mean(input_a), np.std(input_a))
    print('Recon A (mean, std):', np.mean(recon_a), np.std(recon_a))
    # print('Input B (mean, std):', np.mean(input_b), np.std(input_b))
    # print('Recon B (mean, std):', np.mean(recon_b), np.std(recon_b))
    
    # or even
    signal_ex = input_a
    recon_ex = recon_a
    label_ex = label_a
    mysignals = [{'name': 'Original', 'x': np.arange(0,len(signal_ex)),
                 'y': signal_ex, 'color':'k', 'linewidth':7},
                {'name': 'Reconstructed', 'x': np.arange(0,len(signal_ex)),
                 'y': recon_ex, 'color':'b', 'linewidth':3},
                {'name': 'Difference', 'x': np.arange(0,len(signal_ex)),
                 'y': np.abs(signal_ex-recon_ex), 'color':'r', 'linewidth':1}]
    
    fig, ax = plt.subplots()
    for signal in mysignals:
        ax.plot(signal['x'], signal['y'], 
                color=signal['color'], 
                linewidth=signal['linewidth'],
                label=signal['name'])
    
    # Enable legend
    ax.legend()
    ax.set_title("Label = %s" % label_ex)
    plt.show()






























#%% Importing Oncotech data
import pandas as pd
import numpy as np
from skimage.transform import resize

# Define configuration
oncotech_filename = 'Oncotech_Dataset_20250217.csv'
path_to_sets = "RF_dataset/Saved_seeds/VAE_Oncotechonly/"
lesions_class = "Oncotech_Lesion_Class_20250312.csv"

# Selecting parameters [Check before running!]
balance = False
load_data = True # If we have already saved sets
resize_shape = None# This is for 2D. Use int (128?) or None
data_mode = 'RF'

oncotech_file = pd.read_csv(oncotech_filename, delimiter=';') #Open csv as DataFrame
save_string = '/testset_oncotech_%s' % data_mode #Training set pth filename
if resize_shape is not None: save_string += '_dim%i' % resize_shape
if balance: save_string += '_balanced'


# Get some details about the file
oncotech_n_patients = pd.unique(oncotech_file.Patient) # Number of patients with unique IDs
print('In total, we have %i files from %i subjects.' % (len(oncotech_file), len(oncotech_n_patients)))
print('We use all available data from Oncotech as the testing set')

# Preparing test set
if load_data:
    load_string = path_to_sets + save_string + '.pth'
    print('Loading saved training set:', load_string)
    test = torch.load(load_string, weights_only=False)
    
else:
    from scipy.io import loadmat
    all_files = oncotech_file['Filepath']
    images = []
    labels = []
    identifier = []
    print('Using mode:', data_mode)

    for i in range(len(oncotech_file)):
        # Printing progress
        if i % 10 == 0:
            print('Loading file', i, 'of', len(oncotech_file))

        filename = all_files[i]
        RF_data = loadmat(filename)['RFmig']

        for j in range(0, 30, 10): #30 for all frames, or range(1) for just one frame
            single_RF = RF_data[...,j]
            if "-B" in data_mode:
                from utils.oncofunctions import rf2bmode
                norm_RF = rf2bmode(single_RF)
                if resize_shape is not None: norm_RF = resize(norm_RF, (resize_shape, resize_shape))
                images.append(norm_RF)
                labels.append(np.nan)
                identifier.append(filename+'_Frame%s' % (j))
                
            else:
                # Not sure yet if we need to normalize it!
                # Possibilities are: mean 0 + std 1 or ranging between 0-1
                norm_RF = (single_RF - np.mean(single_RF)) / np.std(single_RF)        # mean 0 + std 1
                # norm_RF = (single_RF - np.min(single_RF)); norm_RF /= np.max(norm_RF) #ranging between 0-1
                # norm_RF = single_RF                                                     #not normalizing at all
                
                lines = np.arange(0,norm_RF.shape[1])
                for l in lines:
                    RF_line = norm_RF[:,l]
                    
                    if 'env' in data_mode:
                        from scipy.signal import hilbert
                        envelope = np.abs(hilbert(RF_line))
                        RF_line = envelope
                        
                    if liveronly:
                        print('Selected liver only, option not available right now')
                    
                    images.append(RF_line[:-1]) #remove last point so it's a multiple of 8
                    labels.append(np.nan)
                    identifier.append(filename+'_%s_%s' % (j, l))
                
    X = np.array(images)
    y = np.array(labels)
    z = np.array(identifier)
        
    X = np.expand_dims(X, axis=1) # For the architecture, we expand one dimension to say n_channels=1
    test = [X,y,z]
    print('Saving splits...')
    torch.save(test, path_to_sets + save_string + '.pth', pickle_protocol=4) #Needs protocol >=4 for large files


testloader,_         = prepare_loaders(test, batch_size=1, shuffle=False)


#%%
# If want to visualize the test data
import matplotlib.pyplot as plt
X = test[0]; y = test[1]; z = test[2]
if "2D" in data_mode:
    plt.subplot(1,2,1)
    plt.imshow(X[0,0,...], cmap='gray', aspect='auto')
    plt.title(z[0].split('US_Verasonics\\')[1] + '\nGrade:' + y[0])
    plt.subplot(1,2,2)
    plt.imshow(X[81,0,...], cmap='gray', aspect='auto')
    plt.xlabel(z[81] + '\nGrade:' + str(y[81]))


sample_no = 0 #Patient number
frame_no = 0 #Frame number
### Given that we have 30 frames, and 20 lines per frame
# nframes = 30
# nlines = 20

### Given that we have 3 frames, and all lines
nframes = 3
nlines = 254

sample_npoints = np.arange(sample_no*nframes*nlines,(sample_no+1)*nframes*nlines)
sample_patient = test[0][sample_npoints] # 600 signals in dim (1,3248) (if full signal)

frame_npoints = np.arange(frame_no*nlines, (frame_no+1)*nlines)
frame_RF = sample_patient[frame_npoints]

fig, axes = plt.subplots(5,4)
axes = axes.flatten()
for frame,ax in zip(frame_RF,axes):
    ax.plot(frame[0])
plt.suptitle('Each RF line')
plt.show()

RF_frame = np.transpose(np.squeeze(frame_RF, axis=1))

plt.imshow(RF_frame, cmap='jet', aspect='auto'), plt.title('RF'), plt.show()

from utils.loading_data import rf2bmode as r2b
Bmode_frame = r2b(RF_frame)
plt.imshow(Bmode_frame, cmap='gray', aspect='auto'), plt.title('Bmode'), plt.show()

plt.show()


#%% And visualizing the reconstruction signal
def elbo_loss_2(recon_xray, xray, recon_ecg, ecg, mu, logvar,
              lambda_xray=1.0, lambda_ecg=1.0, annealing_factor=1):

    xray_mse, ecg_mse = 0, 0
    if recon_xray is not None and xray is not None:
        # Reshape to the original image size
        xray_mse = nn.functional.mse_loss(recon_xray, xray, reduction='sum')

    if recon_ecg is not None and ecg is not None:
        # Reshape to the original image size
        ecg_mse = nn.functional.mse_loss(recon_ecg, ecg, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    ELBO = torch.mean(lambda_xray * xray_mse + lambda_ecg * ecg_mse + annealing_factor * KLD)
    
    return ELBO, ecg_mse

def reconstruction_val(model, loader):
    lambda_xray = 0.0
    lambda_ecg = 10.0 # 10
    annealing_factor = 1.0
    
    val_error = []
    val_in = []
    rec_signals = []
    all_labels = []
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for val_batch_idx, (val_inputs, val_labels) in enumerate(loader):
            val_inputs = val_inputs.to(device)
            if val_inputs.shape[2]%2==1: val_inputs = val_inputs[:,:,:-1]
            if 'liveronly' in data_mode:
                print('Using only-liver mode [to be checked]')
                if val_inputs.shape[2] > 400: #liveronly mode
                    downsampling_factor = 4 #Same as Aiguo Han
                    ds_signal = val_inputs[:,:,:-1:downsampling_factor]
                    first_half = ds_signal[:,:, :400]
                    second_half = ds_signal[:,:, 400:800]
                    val_inputs = torch.cat((first_half, second_half), dim=0)
                
            _, val_recon_ecg_only, val_mu_ecg, val_logvar_ecg = model(ecg=val_inputs)
            val_ecg_loss, val_mse = elbo_loss(None, None, val_recon_ecg_only, val_inputs, val_mu_ecg, val_logvar_ecg, lambda_xray, lambda_ecg, annealing_factor)
            # _, val_ecg_loss = elbo_loss_2(None, None, val_recon_ecg_only, val_inputs, val_mu_ecg, val_logvar_ecg, lambda_xray, lambda_ecg, annealing_factor)
            
            val_error.append(val_ecg_loss.item())
            val_in.append(val_inputs.cpu().detach().numpy())
            rec_signals.append(val_recon_ecg_only.cpu().detach().numpy())
            all_labels.append(val_labels)
            
    signal = [v[0] for r in val_in for v in r]
    reconstructed = [s[0] for n in rec_signals for s in n]
    labels = [l for m in all_labels for l in m]
    return signal, reconstructed, labels, val_error

def reprocess_signal(s):
    if type(s) == list:
        #Putting each signal in a list between 0 and 1 and recentering
        a = [i-np.min(i) for i in s]
        b = [i/np.max(i) for i in a]
        c = [i-np.mean(i) for i in b]
    else: 
        print ('Error')
        c=s
    return c

signal, reconstructed, labels, val_error = reconstruction_val(model, testloader)
# signal_rep = reprocess_signal(signal)
# recon_rep = reprocess_signal(reconstructed)

#If testloader uses fullsignal instead of mask
# signal = [np.concatenate((signal[i], signal[i + 1]), axis=0) for i in range(0, len(signal), 2)]
# reconstructed = [np.concatenate((reconstructed[i], reconstructed[i + 1]), axis=0) for i in range(0, len(reconstructed), 2)]

ex = 1981 #Select a number

signal_ex = np.array(signal[ex]) # signal or signal_rep
recon_ex = np.array(reconstructed[ex]) #reconstructed or recon_rep
label_ex = labels[ex]
error_ex = val_error[ex]#/len(signal_ex)
mysignals = [{'name': 'Original', 'x': np.arange(0,len(signal_ex)),
             'y': signal_ex, 'color':'k', 'linewidth':7},
            {'name': 'Reconstructed', 'x': np.arange(0,len(signal_ex)),
             'y': recon_ex, 'color':'b', 'linewidth':3},
            {'name': 'Difference', 'x': np.arange(0,len(signal_ex)),
              # 'y': np.abs(signal_ex-recon_ex), 'color':'r', 'linewidth':1}]
              'y': signal_ex-recon_ex, 'color':'r', 'linewidth':1}]


fig, ax = plt.subplots()
for ex_signal in mysignals:
    ax.plot(ex_signal['x'], ex_signal['y'], 
            color=ex_signal['color'], 
            linewidth=ex_signal['linewidth'],
            label=ex_signal['name'])

# Enable legend
ax.legend(fontsize=18)
# ax.set_ylabel(fontsize=16)
# ax.set_title("Label = %s" % label_ex)
# ax.set_title("Rec error = %.2f" % error_ex )
plt.show()




#%% Looking at the B-mode
def plot_test(signal, sample_no, frame_no, n_frames=3, n_lines = 254):
    sample_npoints = np.arange(sample_no*nframes*nlines,(sample_no+1)*nframes*nlines)
    sample_patient = signal[sample_npoints] # 600 signals in dim (1,3248) (if full signal)
    
    frame_npoints = np.arange(frame_no*nlines, (frame_no+1)*nlines)
    frame_RF = sample_patient[frame_npoints]
    
    fig, axes = plt.subplots(5,4)
    axes = axes.flatten()
    for frame,ax in zip(frame_RF,axes):
        ax.plot(frame)
    plt.suptitle('Each RF line')
    plt.show()
    
    RF_frame = np.transpose(frame_RF)
    plt.imshow(RF_frame, cmap='gray', aspect='auto'), plt.title('RF'), plt.show()
    
    Bmode_frame = r2b(RF_frame)
    #To do: A better processing could improve the visualization -> check matlab scripts.
    plt.imshow(Bmode_frame, cmap='gray', aspect='auto'), plt.title('Bmode'), plt.show()

    return Bmode_frame



sample_no = 0 #Patient number
frame_no = 0 #Frame number
n_frames = 3
n_lines = 254


# signal, reconstructed, labels, val_error = reconstruction_val(model, testloader)
# signal_rep = reprocess_signal(signal)
# recon_rep = reprocess_signal(reconstructed)

in_signal = np.array(signal) # signal or signal_rep
out_signal = np.array(reconstructed) #reconstructed or recon_rep
diff_signal = np.abs(out_signal - in_signal)
 
plot_test(in_signal, sample_no, frame_no)
plot_test(out_signal, sample_no, frame_no)
plot_test(diff_signal, sample_no, frame_no)

diff_repar = diff_signal/diff_signal.max()
diff_recon = diff_repar*255
sample_npoints = np.arange(sample_no*n_frames*n_lines,(sample_no+1)*n_frames*n_lines)
sample_patient = diff_recon[sample_npoints] # 600 signals in dim (1,3248) (if full signal)
frame_npoints = np.arange(frame_no*n_lines, (frame_no+1)*n_lines)
frame_RF = diff_recon[frame_npoints]
RF_frame = np.transpose(frame_RF)
plt.imshow(RF_frame, cmap='gray', aspect='auto'), plt.title('RF'), plt.show()

#%%
import scipy

def extract_features(signal, name, fs=100):  # `fs` is the sampling frequency (Hz)
    """
    Extract features from a 1D signal.

    Parameters:
    - signal: np.ndarray, 1D array representing the signal
    - fs: float, sampling frequency (default=100 Hz)

    Returns:
    - features: dict, extracted features
    """

    # Statistical Features
    features = {
        "mean_"+name: np.nanmean(signal),
        "std_"+name: np.nanstd(signal),
        "max_"+name: np.nanmax(signal),
        "min_"+name: np.nanmin(signal),
        "median_"+name: np.nanmedian(signal),
        "skewness_"+name: scipy.stats.skew(signal, nan_policy='omit'),
        "kurtosis_"+name: scipy.stats.kurtosis(signal, nan_policy='omit'),
    }
    
    # Time-Domain Features
    features.update({
        "rms_"+name: np.sqrt(np.mean(signal**2)),
        "crest_factor_"+name: np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        "zero_crossings_"+name: np.sum(np.diff(np.signbit(signal))),
    })
    
    # Frequency-Domain Features
    freqs, psd = scipy.signal.welch(signal, fs=fs)
    features.update({
        "power_spectral_density_mean_"+name: np.mean(psd),
        "power_spectral_density_std_"+name: np.std(psd),
        "dominant_frequency_"+name: freqs[np.argmax(psd)],
    })
    
    return features

def create_dataset(signals, name, fs=100):
    """
    Create a dataset with features extracted from multiple signals.
    
    Parameters:
    - signals: list of np.ndarray, list of 1D signals
    - fs: float, sampling frequency (default=100 Hz)
    
    Returns:
    - dataset: pd.DataFrame, dataset with features for each signal
    """
    # Initialize a list to store feature dictionaries
    feature_list = []
    
    for signal in signals:
        # Extract features for each signal and add to the list
        features = extract_features(signal, name, fs)
        feature_list.append(features)
    
    # Convert the list of dictionaries to a Pandas DataFrame
    dataset = pd.DataFrame(feature_list)
    return dataset


def saving_all_features(model, mode, loader, subset, save_path="RF_dataset/Saved_seeds/FeaturesResults/"):
        
    if mode == 'val':
        # loader = valloader
        # subset = val
        patient_str = r'VISIT\\QUS_NASH_'
    if mode == 'test':
        # loader = testloader
        # subset = test    
        patient_str = 'ONCOTECH'
    else:
        # loader = trainloader
        # subset = train
        patient_str = r'VISIT\\QUS_NASH_'
    
    print('Reconstructing the signal from %s dataset' % mode)
    signal, reconstructed, labels, error = reconstruction_val(model, loader)
    
    df = pd.DataFrame()
    datapoint_name = pd.Series(subset[-1])
    df['Datapoint'] = list(datapoint_name)
    if mode == 'test': 
        datapoint_name = datapoint_name.str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
    else: 
        datapoint_name = datapoint_name.str.replace("SWE", r"\\", case=False, regex=True)
    a = datapoint_name.str.upper().str.split(patient_str, expand=True)
    b = a.iloc[:,-1].str.split("\\", expand=True)
    patient = b[0] 
    df['Patient'] = list(patient)
    # c = b[b.columns[-1]]  
    d = datapoint_name.str.split('.mat', expand=True)  
    d = d[d.columns[-1]].str.split('_', expand=True)
    frame = d[d.columns[-2]]
    df['Frame'] = list(frame)
    line = d[d.columns[-1]] 
    df['Line'] = list(line)

    df['Label'] = labels
    df['Rec_error'] = error
    df = df.reset_index()

    print('Saving multiple features from different sources...\nCheck extract_features function for details')
    dataset_sig = create_dataset(signal, 'original')
    dataset_rec = create_dataset(reconstructed, 'reconst')
    diff = [signal[i]-reconstructed[i] for i in range(len(signal))]
    dataset_diff = create_dataset(diff, 'diff')
    full_features = pd.concat([df, dataset_sig, dataset_rec, dataset_diff], axis=1)
    full_features.to_csv(save_path+'MultiFeatures_CARDIOVAE_{}_{}_seed-{}-{}.csv'.format(biomarker, data_mode, seed, mode))
    print('Dataframe saved to .csv')
    return full_features, diff









#%% looping across all seeds

for seed in [9, 19, 21, 25, 42]:
    print('Using seed:', seed)
    path_to_sets = "RF_dataset/Saved_seeds/VAE_NASHonly/"
    val = torch.load(path_to_sets + 'val_{}_seed-{}_fullsignal.pth'.format(data_mode, int(seed)), weights_only=False)
    model = CardioVAE(image_input_channels=1, ecg_input_dim=val[0].shape[-1],latent_dim=n_latents).to(device)
    model.load_state_dict(torch.load(path_to_sets+'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed), weights_only=True))
    print('Model loaded:', 'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed))

    path_to_sets = "RF_dataset/Saved_seeds/VAE_Oncotechonly/"
    save_string = '/testset_oncotech_%s' % data_mode #Training set pth filename
    test = torch.load(path_to_sets + save_string + '.pth', weights_only=False)
    
    valloader,_         = prepare_loaders(val, batch_size=1, shuffle=False)
    testloader,_         = prepare_loaders(test, batch_size=1, shuffle=False)
    print('Dataloaders ready.')

    for mode, loader, subset in zip(['val', 'test'],[valloader, testloader], [val, test]):
        ff, d = saving_all_features(model, mode, loader, subset, save_path="RF_dataset/Saved_seeds/FeaturesResults/")
        print('Features saved for %s mode' % mode)

for seed in [0]:
    print('Using seed: %i - Training only on volunteers' % seed)
    path_to_sets = "RF_dataset/Saved_seeds/VAE_NASHonly/"
    val = torch.load(path_to_sets + 'val_{}_seed-{}_Patients_fullsignal.pth'.format(data_mode, int(seed)), weights_only=False)
    model = CardioVAE(image_input_channels=1, ecg_input_dim=train[0].shape[-1],latent_dim=n_latents).to(device)
    model.load_state_dict(torch.load(path_to_sets+'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed), weights_only=True))
    print('Model loaded:', 'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed))

    path_to_sets = "RF_dataset/Saved_seeds/VAE_Oncotechonly/"
    save_string = '/testset_oncotech_%s' % data_mode #Training set pth filename
    test = torch.load(path_to_sets + save_string + '.pth', weights_only=False)
    
    valloader,_         = prepare_loaders(val, batch_size=1, shuffle=False)
    testloader,_         = prepare_loaders(test, batch_size=1, shuffle=False)
    print('Dataloaders ready.')

    for mode, loader, subset in zip(['val', 'test'],[valloader, testloader], [val, test]):
        ff, d = saving_all_features(model, mode, loader, subset, save_path="RF_dataset/Saved_seeds/FeaturesResults/")
        print('Features saved for %s mode' % mode)


#%% Comparing val x test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, mannwhitneyu

data_mode = "RF" # only RF for now
biomarker = "NAS_class"
# save_path="RF_dataset/Saved_seeds/FeaturesResults/"
save_path="RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/"
csv_file = "QUSinNASH_Dataset_20250217.csv"

import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)

lesions_class = "Oncotech_Lesion_Class_20250312.csv"
lesion_classes = pd.read_csv(lesions_class, delimiter=';')
lesion_classes_name = lesion_classes['patientID'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
lesion_classes_name = lesion_classes_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = lesion_classes_name.str.upper().str.split('ONCOTECH', expand=True)[1]
lesion_classes['Patient'] = a

results_re = []
results_t = []

val_list = []
test_list = []

for seed in [0,9,19,21,25,42]:
    print(seed)
    val_df1 = pd.read_csv(save_path+'MultiFeatures_CARDIOVAE_{}_{}_seed-{}-{}.csv'.format(biomarker, data_mode, seed, 'val'))
    test_df = pd.read_csv(save_path+'MultiFeatures_CARDIOVAE_{}_{}_seed-{}-{}.csv'.format(biomarker, data_mode, seed, 'test'))

    data = pd.read_csv(csv_file, delimiter=',')

    if seed == 100:
        train = data.loc[data['Steatosis_percentage'] == "No biopsy"]
        val = data.loc[data['Steatosis_percentage'] != "No biopsy"]
        print('Using only volunteers for training: %i subjects' % len(pd.unique(train['Patient'])))
        print('Using patients for validation: %i subjects' % len(pd.unique(val['Patient'])))
    else:
        data = data.dropna(subset=[biomarker])
        patients_df = data.drop_duplicates(subset=['Patient'])
        train, val = train_test_split(patients_df, test_size=0.25, random_state=seed)
        print('Number of files (included):', data.shape[0])
        print('Number of patients:', patients_df.shape[0])

    trainset = data[data['Patient'].isin(train['Patient'])]
    valset = data[data['Patient'].isin(val['Patient'])]
    print('Train set:', trainset.shape[0], 'files. | Unique patients:', trainset['Patient'].nunique())
    print('Validation set:', valset.shape[0], 'files. | Unique patients:', valset['Patient'].nunique())

    training_pats = set(trainset['Patient'].str.split('_', expand=True)[2])    
    val_pats = set(val_df1['Patient'].str.lower().str.split('_', expand=True)[0]) - training_pats
    if len(val_pats) != len(pd.unique(val_df1['Patient'])):
        val_df = val_df1.loc[val_df1['Patient'].isin(val_pats)]
        print('Only %i val patients remaining' % len(pd.unique(val_df['Patient'])))
        print(pd.unique(val_df['Patient']))
    else:
        val_df = val_df1
        
    if len(pd.unique(val_df['Patient']))==0:
        print('No single patients on this valset')
        continue

    val_list.append(val_df)
    test_list.append(test_df)
    
    #Add the lesions GT
    final_oncotech = pd.merge(lesion_classes, test_df)
    
    
    columns = val_df.columns
    metrics = columns[7:]
    
    rec_error = metrics[0]
    metrics_og = metrics[1:14]
    metrics_rec = metrics[14:27]
    metrics_diff = metrics[27:]
    
    # Group by patients
    # val_grouped = val_df.groupby(['Patient']).mean(numeric_only=True)
    # val_grouped_std = val_df.groupby(['Patient']).std(numeric_only=True)
    # test_grouped = test_df.groupby(['Patient']).mean(numeric_only=True)
    # test_grouped_std = test_df.groupby(['Patient']).std(numeric_only=True)
    
    # Group by file
    val_df['Filepath'] = val_df['Datapoint'].str.split(r'1st_visit\\', expand=True)[1].str.split(r'\\QUS_', expand=True)[0]
    test_df['Filepath'] = test_df['Datapoint'].str.split(r'US_Verasonics\\', expand=True)[1].str.split(r'\\QUS_', expand=True)[0]
    val_grouped = val_df.groupby(['Filepath']).mean(numeric_only=True)
    val_grouped_std = val_df.groupby(['Filepath']).std(numeric_only=True)
    test_grouped = test_df.groupby(['Filepath']).mean(numeric_only=True)
    test_grouped_std = test_df.groupby(['Filepath']).std(numeric_only=True)
    
    comparison_df = pd.DataFrame({
        "val_mean": val_grouped.mean(),
        "val_std": val_grouped_std.mean(),
        "test_mean": test_grouped.mean(),
        "test_std": test_grouped_std.mean()
    })
    
    # print(comparison_df)
      
    results = {}
    for metric in metrics:
        t_stat, p_value = ttest_ind(val_grouped[metric], test_grouped[metric], equal_var=False)  # Welch's t-test
        u_stat, p_value_u = mannwhitneyu(val_grouped[metric], test_grouped[metric], alternative='two-sided')
    
        results[metric] = {
            "t-test_p": p_value,
            "mannwhitney_p": p_value_u
        }
    
    results_df = pd.DataFrame(results).T
    # print(results_df)
    filtered_results = results_df[(results_df["t-test_p"] < 0.05)] #| (results_df["mannwhitney_p"] < 0.05)]
    # print(filtered_results)
    
    alpha_bonferroni = 0.05 / 40 # Rec_error + 3*13 features | or 0.05 len(results_df)  
    filtered_bonferroni = results_df[
        (results_df["t-test_p"] < alpha_bonferroni)]# & 
    #     (results_df["mannwhitney_p"] < alpha_bonferroni)
    # ]
    print(filtered_bonferroni)
    
    # results_df["t-test_p_bonf"] = results_df["t-test_p"] * len(results_df)
    # results_df["mannwhitney_p_bonf"] = results_df["mannwhitney_p"] * len(results_df)
    # print(results_df)
    results_re.append(comparison_df)
    results_t.append(filtered_results)

    #Plot errorbars
    import matplotlib.pyplot as plt
    bar_colors = ['green', 'red']  # Blue and Orange for better contrast
    # Define x positions
    x_pos = [1, 2]
    # Get values
    val_mean = comparison_df.loc['Rec_error']['val_mean']
    test_mean = comparison_df.loc['Rec_error']['test_mean']
    val_std = comparison_df.loc['Rec_error']['val_std']#/len(val_grouped)
    test_std = comparison_df.loc['Rec_error']['test_std']#/len(test_grouped)
    # Create figure
    plt.figure(figsize=(6, 5))
    # Plot bars
    plt.bar(x_pos[0], val_mean, color=bar_colors[0], width=0.6, label="Validation")
    plt.bar(x_pos[1], test_mean, color=bar_colors[1], width=0.6, label="Test")
    # Add error bars
    # plt.errorbar(x_pos[0], val_mean, yerr=val_std, fmt='none', capsize=5, capthick=2, color='black')
    # plt.errorbar(x_pos[1], test_mean, yerr=test_std, fmt='none', capsize=5, capthick=2, color='black')
    # Labels and title
    plt.xticks(x_pos, ['Without lesions', 'With lesions'], fontsize=12)
    # plt.ylim([0,np.max(test_std)])
    plt.ylabel("Reconstruction Error", fontsize=14)
    plt.xlabel("Dataset", fontsize=14)
    plt.title("Comparison of Reconstruction Error", fontsize=16)
    # Grid for better readability
    # plt.grid(axis='y', linestyle='--', alpha=0.6)
    # Add legend
    plt.legend(fontsize=12)
    # Show plot
    plt.show()
    t_stat, p_value = ttest_ind(val_grouped['Rec_error'], test_grouped['Rec_error'], equal_var=False)  # Welch's t-test
    print(p_value)

# Error bars for all CVs
import numpy as np
rec_error_cv_val_mean = []
rec_error_cv_test_mean = []
rec_error_cv_val_std = []
rec_error_cv_test_std= []
for i,cv in enumerate(results_re):
    # if i==4:
    #     continue
    rec_error_cv_val_mean.append(cv.loc['Rec_error']['val_mean'])
    rec_error_cv_test_mean.append(cv.loc['Rec_error']['test_mean'])
    rec_error_cv_val_std.append(cv.loc['Rec_error']['val_std'])
    rec_error_cv_test_std.append(cv.loc['Rec_error']['test_std'])
val_mean_re = np.mean(rec_error_cv_val_mean)
val_std_re = np.mean(rec_error_cv_val_std)
test_mean_re = np.mean(rec_error_cv_test_mean)
test_std_re = np.mean(rec_error_cv_test_std)
t_stat, p_value = ttest_ind(rec_error_cv_val_mean, rec_error_cv_test_mean, equal_var=False)

import matplotlib.pyplot as plt
plt.bar(1,val_mean_re, color='black')
plt.bar(2,test_mean_re, color='red')
plt.errorbar(1,val_mean_re, yerr=val_std_re)
plt.errorbar(2,test_mean_re, yerr=test_std_re)
plt.show()

final_val_df = pd.concat(val_list, ignore_index=True)
final_test_df = pd.concat(test_list, ignore_index=True)

final_val_df.to_csv(save_path + 'Combined_Val.csv', index=False)
final_test_df.to_csv(save_path + 'Combined_Test.csv', index=False)

#%% Investigating combined sets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, mannwhitneyu

data_mode = "RF" # only RF for now
biomarker = "NAS_class"
# save_path="RF_dataset/Saved_seeds/FeaturesResults/"
save_path="RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/"
csv_file = "QUSinNASH_Dataset_20250217.csv"

import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)

lesions_class = "Oncotech_Lesion_Class_20250312.csv"
lesion_classes = pd.read_csv(lesions_class, delimiter=';')
lesion_classes_name = lesion_classes['patientID'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
lesion_classes_name = lesion_classes_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = lesion_classes_name.str.upper().str.split('ONCOTECH', expand=True)[1]
lesion_classes['Patient'] = a

print('Loading validation results')
val_combo = pd.read_csv(save_path+'Combined_Val.csv')
print('Loading test results')
test_combo = pd.read_csv(save_path+'Combined_Test.csv')

test_combo_name = test_combo['Datapoint'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
test_combo_name = test_combo_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = test_combo_name.str.upper().str.split('ONCOTECH', expand=True)[2]
b = a.str.split(r'\\', expand=True)[0]
test_combo['Patient'] = b

columns = val_combo.columns
metrics = columns[7:]

# Group by patients
# val_grouped = val_combo.groupby(['Patient']).mean(numeric_only=True)
# val_grouped_std = val_combo.groupby(['Patient']).std(numeric_only=True)
# test_grouped = test_combo.groupby(['Patient']).mean(numeric_only=True)
# test_grouped_std = test_combo.groupby(['Patient']).std(numeric_only=True)

# Group by file
val_combo['Filepath'] = val_combo['Datapoint'].str.split(r'1st_visit\\', expand=True)[1].str.split(r'\\QUS_', expand=True)[0]
test_combo['Filepath'] = test_combo['Datapoint'].str.split(r'US_Verasonics\\', expand=True)[1].str.split(r'\\QUS_', expand=True)[0]
val_grouped = val_combo.groupby(['Filepath']).mean(numeric_only=True)
val_grouped_std = val_combo.groupby(['Filepath']).std(numeric_only=True)
test_grouped = test_combo.groupby(['Filepath']).mean(numeric_only=True)
test_grouped_std = test_combo.groupby(['Filepath']).std(numeric_only=True)

# val_grouped.to_csv(save_path+'Combined_Val_Grouped_Files.csv')
# test_grouped.to_csv(save_path+'Combined_Test_Grouped_Files.csv')

comparison_df = pd.DataFrame({
    "val_mean": val_grouped.mean(),
    "val_std": val_grouped_std.mean(),
    "test_mean": test_grouped.mean(),
    "test_std": test_grouped_std.mean()
})
print('Results grouped')

t_stat, p_value = ttest_ind(val_grouped['Rec_error'], test_grouped['Rec_error'], equal_var=False)  # Welch's t-test
u_stat, p_value_u = mannwhitneyu(val_grouped['Rec_error'], test_grouped['Rec_error'], alternative='two-sided')

#Plot errorbars
import matplotlib.pyplot as plt
bar_colors = ['green', 'red']  # Blue and Orange for better contrast
# Define x positions
x_pos = [1, 2]
# Get values
val_mean = comparison_df.loc['Rec_error']['val_mean']
test_mean = comparison_df.loc['Rec_error']['test_mean']
val_std = comparison_df.loc['Rec_error']['val_std']#/len(val_grouped)
test_std = comparison_df.loc['Rec_error']['test_std']#/len(test_grouped)
# Create figure
plt.figure(figsize=(6, 5))
# Plot bars
plt.bar(x_pos[0], val_mean, color=bar_colors[0], width=0.6, label="Validation")
plt.bar(x_pos[1], test_mean, color=bar_colors[1], width=0.6, label="Test")
# Add error bars
# plt.errorbar(x_pos[0], val_mean, yerr=val_std, fmt='none', capsize=5, capthick=2, color='black')
# plt.errorbar(x_pos[1], test_mean, yerr=test_std, fmt='none', capsize=5, capthick=2, color='black')
# Labels and title
plt.xticks(x_pos, ['Without lesions', 'With lesions'], fontsize=12)
# plt.ylim([0,np.max(test_std)])
plt.ylabel("Reconstruction Error", fontsize=14)
plt.xlabel("Dataset", fontsize=14)
plt.title("Comparison of Reconstruction Error", fontsize=16)
# Grid for better readability
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# Add legend
plt.legend(fontsize=12)
# Show plot
plt.show()
t_stat, p_value = ttest_ind(val_grouped['Rec_error'], test_grouped['Rec_error'], equal_var=False)  # Welch's t-test
rec_ratio = np.mean(test_grouped['Rec_error'])/np.mean(val_grouped['Rec_error'])
print('Reconstruction error is %.2f times bigger for lesions (%.3f)' % (rec_ratio, p_value))
print('QUS in NASH: %.2f // Oncotech: %.2f' % (np.mean(val_grouped['Rec_error'])/3248, np.mean(test_grouped['Rec_error'])/3248))
### Note: I am using division by 3248 because that is the length of the signal and I want smaller numbers

# AUC
from sklearn.metrics import auc
import numpy as np

from scipy.stats import norm
def delong_confidence_interval(y_true, y_scores, alpha=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Get standard error using DeLong’s method
    n1 = sum(y_true == 1)  # Number of positive samples
    n2 = sum(y_true == 0)  # Number of negative samples
    
    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc**2 / (1 + roc_auc)
    
    se = np.sqrt((roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc**2) + (n2 - 1) * (q2 - roc_auc**2)) / (n1 * n2))
    
    # Compute the confidence interval
    z = norm.ppf((1 + alpha) / 2)  # 1.96 for 95% CI
    lower_bound = roc_auc - z * se
    upper_bound = roc_auc + z * se
    
    return roc_auc, (max(0, lower_bound), min(1, upper_bound))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# #For MASH vs Onco
data_0 = val_grouped
data_1 = test_grouped
#For Benign vs Malig (Use test_grouped by patient!)
# final_oncotech = pd.merge(lesion_classes, test_grouped.reset_index())
# data_0 = final_oncotech.loc[final_oncotech['Malign']==0]
# data_1 = final_oncotech.loc[final_oncotech['Malign']==1]

# Get data
y_val = np.zeros(len(data_0))
y_test = np.ones(len(data_1))  
# Concatenate Rec_error values and labels
y_true = np.concatenate([y_val, y_test])
y_scores = np.concatenate([data_0['Rec_error'], data_1['Rec_error']])
# Compute ROC curve
fpr, tpr, th = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
# Plot
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f"AUC = {roc_auc:.3f}")
plt.fill_between(fpr, tpr, alpha=0.1, color='red')  # Shaded area
# Plot diagonal reference line
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
# Labels and Title
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curve - No lesion vs lesions", fontsize=16)
# Grid and legend
plt.grid(alpha=0.3)
plt.legend(loc="lower right", fontsize=12)
plt.show()

youden_index = tpr - fpr
best_threshold = th[np.argmax(youden_index)]
y_pred = (y_scores >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)  # Recall
specificity = tn / (tn + fp)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")


auc_value, auc_ci = delong_confidence_interval(y_true, y_scores)
print(f"AUC: {auc_value:.3f}, 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")


t_stat, p_value = ttest_ind(data_0['Rec_error'], data_1['Rec_error'], equal_var=False)  # Welch's t-test
print('Data 0: %.2f // Data 1: %.2f' % (np.mean(data_0['Rec_error'])/3248, np.mean(data_1['Rec_error'])/3248))



# Box plots
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
plt.figure()
bpl = plt.boxplot(data_0['Rec_error'], positions=[0], sym='', widths=0.6)
bpr = plt.boxplot(data_1['Rec_error'], positions=[1], sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Validation - QUS in NASH')
plt.plot([], c='#2C7BB6', label='Test - Oncotech')
plt.legend()

ticks = ['QUS in NASH', 'Oncotech']
plt.xticks(np.arange(0, len(ticks)), ticks, rotation=45)
# plt.xlim(-2, len(ticks)*2)
# plt.ylim(-0.01, 0.01)
plt.tight_layout()
plt.show()


#%% Getting the results already grouped
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)

#Plots function
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
def boxplot_multi(data_array, names):
    colorlist = ['#2ca25f','#D7191C','#2C7BB6','#feb24c','#8856a7'] # colors are from http://colorbrewer2.org/
    colors = colorlist[:len(data_array)]
    
    fig, ax = plt.subplots()
    i=0
    for don, nom, cor in zip(data_array,names,colors):
        bp = ax.boxplot(don, positions=[i], sym='', widths=0.5, patch_artist=True)
        set_box_color(bp, cor) 
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(cor)
            patch.set_alpha(0.5)
        i += 1
        plt.plot([], c=cor, label=nom)
    plt.legend()
    
    
    plt.xlabel("Tested group", fontsize=14)
    plt.ylabel("Reconstruction error (a.u.)", fontsize=14)
    # plt.title("Boxplots", fontsize=16)
    # Grid and legend
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=12)
    plt.ylim([0, 3.7])
    
    plt.xticks([])#np.arange(0, len(names)), names, rotation=45)
    plt.tight_layout()
    plt.show()
    
from scipy.stats import norm
def delong_confidence_interval(y_true, y_scores, alpha=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Get standard error using DeLong’s method
    n1 = sum(y_true == 1)  # Number of positive samples
    n2 = sum(y_true == 0)  # Number of negative samples
    
    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc**2 / (1 + roc_auc)
    
    se = np.sqrt((roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc**2) + (n2 - 1) * (q2 - roc_auc**2)) / (n1 * n2))
    
    # Compute the confidence interval
    z = norm.ppf((1 + alpha) / 2)  # 1.96 for 95% CI
    lower_bound = roc_auc - z * se
    upper_bound = roc_auc + z * se
    
    return roc_auc, (max(0, lower_bound), min(1, upper_bound))

#AUC function
def auc_metrics(data_array, names):
    # Get data
    y_val = np.zeros(len(data_array[0]))
    y_test = np.ones(len(data_array[1]))  
    # Concatenate Rec_error values and labels
    y_true = np.concatenate([y_val, y_test])
    y_scores = np.concatenate([data_array[0], data_array[1]])
    # Compute ROC curve
    fpr, tpr, th = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.fill_between(fpr, tpr, alpha=0.1, color='blue')  # Shaded area
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
    # Labels and Title
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve - No lesion vs lesions", fontsize=16)
    # Grid and legend
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()
    
    youden_index = tpr - fpr
    best_threshold = th[np.argmax(youden_index)]
    y_pred = (y_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc_value, auc_ci = delong_confidence_interval(y_true, y_scores)
    print("Results for %s vs %s" % (names[0], names[1]))
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc_value:.3f}, 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")

data_mode = "RF" # only RF for now
biomarker = "NAS_class"
# save_path="RF_dataset/Saved_seeds/FeaturesResults/"
save_path="RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/"
csv_file = "QUSinNASH_Dataset_20250217.csv"

print('Loading validation results')
val_combo = pd.read_csv(save_path+'Combined_Val_Grouped_Files.csv')
data_val = val_combo['Rec_error']
print('Loading test results')

#### If using Combined_Test_Grouped_Files
test_combo = pd.read_csv(save_path+'Combined_Test_Grouped_Files.csv')
test_combo_details = test_combo['Filepath'].str.split(r'\\', expand=True)
test_combo['Details'] = test_combo_details[1]
test_combo['Filepath_l'] = test_combo['Filepath'].str.lower()
test_combo_name = test_combo_details[0].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
test_combo_name = test_combo_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = test_combo_name.str.upper().str.split('ONCOTECH', expand=True)[1]
test_combo['Patient'] = a

### If using Combined_Test_LesionDetails
# test_combo = pd.read_csv(save_path+'Combined_Test_LesionDetails.csv', delimiter=';')
# test_combo_details = test_combo['Filepath'].str.split(r'Verasonics\\', expand=True)
# test_combo['Details'] = test_combo_details[1].str.split(r'\\QUS_migr', expand=True)[0].str.split(r'\\', expand=True)[1]
# test_combo['Filepath_l'] = test_combo_details[1].str.split(r'\\QUS_migr', expand=True)[0].str.lower()
# test_combo_name = test_combo['Filepath_l'].str.split(r'\\',expand=True)[0].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
# test_combo_name = test_combo_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
# a = test_combo_name.str.upper().str.split('ONCOTECH', expand=True)[1]
# test_combo['Patient'] = a


# Getting the malignancy class
lesions_class = "Oncotech_Lesion_Class_20250312.csv"
lesion_classes = pd.read_csv(lesions_class, delimiter=';')
lesion_classes_name = lesion_classes['patientID'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
lesion_classes_name = lesion_classes_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = lesion_classes_name.str.upper().str.split('ONCOTECH', expand=True)[1]
lesion_classes['Patient'] = a#.astype(int)

final_oncotech = pd.merge(lesion_classes, test_combo.reset_index(), on='Patient')
data_0 = val_combo['Rec_error']/3248
data_1 = final_oncotech['Rec_error']/3248
boxplot_multi([data_0, data_1],['Val', 'Test'])
auc_metrics([data_0, data_1],['Val', 'Test'])



# # Including only Arnaud's or Iman's
lesions_2include = "Oncotech_Include_20250321.csv"
to_include = pd.read_csv(lesions_2include, delimiter=';')
to_include['Filepath'] = to_include['patientID']+"\\"+to_include['AcqName']
to_include['Filepath_l'] = to_include['Filepath'].str.lower()
to_include_name = to_include['patientID'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
to_include_name = to_include_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = to_include_name.str.upper().str.split('ONCOTECH', expand=True)[1]
to_include['Patient'] = a
include_a = to_include.loc[to_include['Arnaud']==1]
include_i = to_include.loc[to_include['Iman']==1]

included_A = pd.merge(test_combo,include_a,on='Filepath_l')
included_A = included_A.rename(columns={'Patient_y':'Patient'})
missing = set(include_a['patientID']) - set(included_A['patientID'])
included_I = pd.merge(test_combo,include_i,on='Filepath_l')
included_I = included_I.rename(columns={'Patient_y':'Patient'})
missing = set(include_i['patientID']) - set(included_I['patientID'])
data_0 = included_A['Rec_error']
data_1 = included_I['Rec_error']
boxplot_multi([data_val, data_0, data_1],['Val', 'Test - Arnauds', 'Test - Imans'])
auc_metrics([data_val, data_0],['Val', 'Test - Arnauds'])
auc_metrics([data_val, data_1],['Val', 'Test - Imans'])


#Selecting by view:
sagittal_df = final_oncotech.loc[final_oncotech['Details'].str.lower().str.contains('sag')]
trans_df = final_oncotech.loc[final_oncotech['Details'].str.lower().str.contains('trans')]
rest = final_oncotech.loc[~final_oncotech['Details'].str.lower().str.contains('trans')]
rest = rest.loc[~rest['Details'].str.lower().str.contains('sag')]
data_0 = sagittal_df['Rec_error']
data_1 = trans_df['Rec_error']
# data_array = [val_combo['Rec_error'], data_0, data_1]
# names = ['Val', 'Test - Sagittal', 'Test - Transverse']
boxplot_multi([data_val, data_0, data_1],['Val', 'Test - Sagittal', 'Test - Transverse'])
auc_metrics([data_val, data_0],['Val', 'Test - Sagittal'])
auc_metrics([data_val, data_1],['Val', 'Test - Transverse'])

#Selecting by malignancy
data_0 = final_oncotech.loc[final_oncotech['Malign']==0]
data_1 = final_oncotech.loc[final_oncotech['Malign']==1]
# data_0 = sagittal_df.loc[sagittal_df['Malign']==0]
# data_1 = sagittal_df.loc[sagittal_df['Malign']==1]


# Using Iman's + Sagittal
data_X = included_I.loc[included_I['Details'].str.lower().str.contains('sag')]
boxplot_multi([data_val, data_X['Rec_error']],['Val', 'Test - Iman and Sagittal'])
auc_metrics([data_val, data_X['Rec_error']],['Val', 'Test - Iman and Sagittal'])


# Lesion size and depth OR B-mode CNR
lesions_dets = "Oncotech_Lesion_details_20250321.csv"
deets = pd.read_csv(lesions_dets, delimiter=';')
deets['Filepath'] = deets['patientID']+"\\"+deets['AcqName']
deets['Filepath_l'] = deets['Filepath'].str.lower()
deets_name = deets['patientID'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
deets_name = deets_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = deets_name.str.upper().str.split('ONCOTECH', expand=True)[1]
deets['Patient'] = a
final_lesions = pd.merge(deets, final_oncotech, on='Filepath_l')

# Chose Lesion_depth, Lesion_size or CNR_Bmode
# Mean thresholds are 5.6, 2.7 and 0.33
th = 5
data_0 = final_lesions.loc[final_lesions['Lesion_depth']<=th]['Rec_error']
data_1 = final_lesions.loc[final_lesions['Lesion_depth']>th]['Rec_error']
boxplot_multi([data_val, data_0, data_1],['Val', 'Test - Smol', 'Test - Biggie'])
auc_metrics([data_val, data_0],['Val', 'Test - Smol'])
auc_metrics([data_val, data_1],['Val', 'Test - Biggie'])
auc_metrics([data_0, data_1],['Test - Smol', 'Test - Biggie'])


# Big, shallow lesions. Malign x Benign
df_0 = final_lesions.loc[final_lesions['Lesion_size']>2.7]
df_0 = df_0.loc[df_0['Lesion_depth']<5.6]
data_0 = df_0.loc[df_0['Malign_x']==0]['Rec_error']
data_1 = df_0.loc[df_0['Malign_x']==1]['Rec_error']
boxplot_multi([data_val, data_0, data_1],['Val', 'Test - Benign', 'Test - Malignant'])
auc_metrics([data_val, data_0],['Val', 'Test - Benign'])
auc_metrics([data_val, data_1],['Val', 'Test - Malignant'])
auc_metrics([data_0, data_1],['Test - Benign', 'Test - Malignant'])



#%% Getting the values within the lesions
import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from skimage.transform import resize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

runcell('Model and Training functions', '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code/1D_VAE_MASHtoOnco.py')
mode = 'val'

# Define configuration
oncotech_filename = 'Oncotech_Dataset_20250217.csv'
path_to_sets = "RF_dataset/Saved_seeds/VAE_Oncotechonly/"
lesions_class = "Oncotech_Lesion_Class_20250312.csv"
path_to_weights = 'RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/'

# Selecting parameters [Check before running!]
balance = False
load_data = True # If we have already saved sets
resize_shape = None# This is for 2D. Use int (128?) or None
data_mode = 'RF'
biomarker = 'NAS_class'

oncotech_file = pd.read_csv(oncotech_filename, delimiter=';') #Open csv as DataFrame
save_string = '/testset_oncotech_%s' % data_mode #Training set pth filename
if resize_shape is not None: save_string += '_dim%i' % resize_shape
if balance: save_string += '_balanced'
onco_expanded = oncotech_file.loc[np.repeat(oncotech_file.index, 3)].reset_index(drop=True)


# Get some details about the file
oncotech_n_patients = pd.unique(oncotech_file.Patient) # Number of patients with unique IDs
print('In total, we have %i files from %i subjects.' % (len(oncotech_file), len(oncotech_n_patients)))
print('We use all available data from Oncotech as the testing set')

# Preparing test set
if load_data:
    load_string = path_to_sets + save_string + '.pth'
    print('Loading saved training set:', load_string)
    test = torch.load(load_string, weights_only=False)
    
    
n_frames = 3
n_lines = 254
n_latents = 256

assert len(oncotech_file) == len(test[2])/n_frames/n_lines

onco_signals = test[0]

def prepare_loaders(data, batch_size=32, shuffle=False, zeroesonly=False):
    X, y, z = data
    if zeroesonly:
        idx = np.where(y==0)[0]
        X, y, z = X[idx], y[idx], z[idx]
        
    X = X.astype(np.float32)
    # y = y.astype(np.int64)
    dataset = list(zip(X, y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, z

testloader,_         = prepare_loaders(test, batch_size=1, shuffle=False)

def acquire_mask(filename):
    # mask_file = filename[:-4] #Remove extension
    # mask_file = mask_file + '_EchoEnv.dat_mask2.dat'
    
    mask_file = filename
    
    fid = open(mask_file, 'r')
    size_Mask = np.fromfile(fid, np.int32,4)
    mask = np.fromfile(fid, np.int8,-1)
    mask = mask[:30*254*3249] #to avoid one patient with 100 frames
    mask2 = mask.reshape(30,254,3249).T
    return mask2

def reconstruction_val(model, loader):
    lambda_xray = 0.0
    lambda_ecg = 10.0 # 10
    annealing_factor = 1.0
    
    val_error = []
    val_in = []
    rec_signals = []
    all_labels = []
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for val_batch_idx, (val_inputs, val_labels) in enumerate(loader):
            val_inputs = val_inputs.to(device)
            if val_inputs.shape[2]%2==1: val_inputs = val_inputs[:,:,:-1]
            if 'liveronly' in data_mode:
                print('Using only-liver mode [to be checked]')
                if val_inputs.shape[2] > 400: #liveronly mode
                    downsampling_factor = 4 #Same as Aiguo Han
                    ds_signal = val_inputs[:,:,:-1:downsampling_factor]
                    first_half = ds_signal[:,:, :400]
                    second_half = ds_signal[:,:, 400:800]
                    val_inputs = torch.cat((first_half, second_half), dim=0)
                
            _, val_recon_ecg_only, val_mu_ecg, val_logvar_ecg = model(ecg=val_inputs)
            val_ecg_loss, mse = elbo_loss(None, None, val_recon_ecg_only, val_inputs, val_mu_ecg, val_logvar_ecg, lambda_xray, lambda_ecg, annealing_factor)
            # _, val_ecg_loss = elbo_loss_2(None, None, val_recon_ecg_only, val_inputs, val_mu_ecg, val_logvar_ecg, lambda_xray, lambda_ecg, annealing_factor)
            
            val_error.append(val_ecg_loss.item())
            val_in.append(val_inputs.cpu().detach().numpy())
            rec_signals.append(val_recon_ecg_only.cpu().detach().numpy())
            all_labels.append(val_labels)
            
    signal = [v[0] for r in val_in for v in r]
    reconstructed = [s[0] for n in rec_signals for s in n]
    labels = [l for m in all_labels for l in m]
    return signal, reconstructed, labels, val_error


def extract_features2D(signal, name):  # `fs` is the sampling frequency (Hz)
    """
    Extract features from a 1D signal.

    Parameters:
    - signal: np.ndarray, 1D array representing the signal
    - fs: float, sampling frequency (default=100 Hz)

    Returns:
    - features: dict, extracted features
    """
    import scipy
    if type(signal) == list:
        signal = np.array(signal)
    
    skewness_per_image = np.array([
        scipy.stats.skew(img[~np.isnan(img)], bias=False) if np.any(~np.isnan(img)) else np.nan
        for img in signal
    ])

    kurtosis_per_image = np.array([
        scipy.stats.kurtosis(img[~np.isnan(img)], bias=False) if np.any(~np.isnan(img)) else np.nan
        for img in signal
    ])
    
    # Statistical Features
    features = {
        "mean_"+name: np.nanmean(signal, axis=(1,2)),
        "std_"+name: np.nanstd(signal, axis=(1,2)),
        "max_"+name: np.nanmax(signal, axis=(1,2)),
        "min_"+name: np.nanmin(signal, axis=(1,2)),
        "median_"+name: np.nanmedian(signal, axis=(1,2)),
        "skewness_"+name: skewness_per_image,
        "kurtosis_"+name: kurtosis_per_image,
    }
       
    return features

if mode=='test':
    all_masks = oncotech_file['Maskpath']
    images = []
    sizes = []
    print('Using mode:', data_mode)
    for i in range(len(all_masks)):
        # Printing progress
        if i > 0 and i % 10 == 0:
            print('Loading file', i, 'of', len(all_masks))
        
        mask = acquire_mask(all_masks[i])
        for j in range(0, 30, 10):
            single_mask = mask[:-1,:,j] #removing the last line of the 2D image
            lesion_size = np.nansum(single_mask)
            
            images.append(single_mask)
            sizes.append(lesion_size)
    
    all_lesions_masks = np.array(images)

for seed in [0, 9, 19, 21, 25, 42]:
    print('Using seed:', seed)
    model = CardioVAE(image_input_channels=1, ecg_input_dim=onco_signals.shape[-1],latent_dim=n_latents).to(device)
    model.load_state_dict(torch.load(path_to_weights+'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed), weights_only=True))
    print('Model loaded:', 'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed))


    signal, reconstructed, labels, val_error = reconstruction_val(model, testloader)
    # signal_rep = reprocess_signal(signal)
    # recon_rep = reprocess_signal(reconstructed)
    # diff_sig = np.abs(np.array(reconstructed)-np.array(signal))
    diff_sig = np.array(signal)

    img_array_shape = [len(oncotech_file)*n_frames, n_lines, len(signal[0])]
    diff_imgs = diff_sig.reshape(img_array_shape).transpose(0, 2, 1)

    if mode == 'test':
        assert diff_imgs.shape == all_lesions_masks.shape
        
        inside_mask = np.where(all_lesions_masks == 1, diff_imgs, np.nan) 
        outside_mask = np.where(all_lesions_masks == 0, diff_imgs, np.nan)
    
        in_feats = extract_features2D(inside_mask, 'lesion')
        out_feats = extract_features2D(outside_mask, 'outside')
        
        features_list = [out_feats, in_feats]    
        dataset = pd.DataFrame({**out_feats, **in_feats})
        assert len(dataset) == len(onco_expanded)
        final_results = pd.concat([onco_expanded, dataset], axis=1)
        
    else:
        feats = extract_features2D(diff_imgs, 'outside')
        dataset = pd.DataFrame(feats)
        final_results = dataset
    
    # final_results.to_csv(path_to_weights+'FeaturesResults/'+'Oncotech_Lesion_InOut_Comparison_{}.csv'.format(seed))
    final_results.to_csv(path_to_weights+'FeaturesResults/'+'Oncotech_InputSignals_{}.csv'.format(seed))


########### Merging dataframes    
'''
import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)
import pandas as pd
path_to_results = 'RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/'
reslist = []
for seed in [9, 19, 21, 25, 42]:
    lesions_results = pd.read_csv(path_to_results+'Oncotech_Lesion_InOut_Comparison_{}.csv'.format(seed)).reset_index()
    reslist.append(lesions_results)
    
final_df = pd.concat(reslist, ignore_index=True)

grouped = final_df.groupby(['Filepath']).mean(numeric_only=True)
grouped = grouped.drop(columns=['index','Unnamed: 0']).reset_index()

files_data = grouped['Filepath'].str.split(r'US_Verasonics\\', expand=True)[1]
files_data = files_data.str.split(r'\\QUS_migr', expand=True)[0]

patients = files_data.str.split(r'\\',expand=True)[0]
acquisition = files_data.str.split(r'\\',expand=True)[1]

patients_rename = patients.str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
patients_rename = patients_rename.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = patients_rename.str.upper().str.split('ONCOTECH', expand=True)[1]
grouped['patientID'] = a
grouped['Notes'] = acquisition

grouped.to_csv(path_to_results + 'Combined_Test_LesionDetails_.csv', index=False)    
  '''  
    
    

#%% Results for in/out lesions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import roc_curve, auc, confusion_matrix

data_mode = "RF" # only RF for now
biomarker = "NAS_class"
# save_path="RF_dataset/Saved_seeds/FeaturesResults/"
save_path="RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/"
csv_file = "QUSinNASH_Dataset_20250217.csv"

import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
def boxplot_multi(data_array, names):
    colorlist = ['#2ca25f','#D7191C','#2C7BB6','#feb24c','#8856a7'] # colors are from http://colorbrewer2.org/
    colors = colorlist[:len(data_array)]
    
    plt.figure()
    i=0
    for don, nom, cor in zip(data_array,names,colors):
        bp = plt.boxplot(don, positions=[i], sym='', widths=0.5)
        set_box_color(bp, cor) 
        i += 1
    
    # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=cor, label=nom)
    plt.legend()
        
    plt.xticks(np.arange(0, len(names)), names, rotation=45)
    plt.tight_layout()
    plt.show()

lesions_class = "Oncotech_Lesion_Class_20250312.csv"
lesion_classes = pd.read_csv(lesions_class, delimiter=';')
lesion_classes_name = lesion_classes['patientID'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
lesion_classes_name = lesion_classes_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = lesion_classes_name.str.upper().str.split('ONCOTECH', expand=True)[1]
lesion_classes['patientID'] = a

print('Loading validation results')
# val_combo = pd.read_csv(save_path+'NASH_NoLesion_Out_Comparison_GroupedSansTrain.csv', delimiter=',') 
val_combo = pd.read_csv(save_path+'NASH_NoLesion_Input_Comparison_GroupedSansTrain.csv', delimiter=',') 
val_grouped = val_combo.groupby('Acquisition').mean().reset_index()
print('Loading test results')
# test_combo = pd.read_csv(save_path + 'Combined_Test_LesionDetails.csv', delimiter=';')    
test_combo = pd.read_csv(save_path + 'Combined_Test_LesionDetails_InputSigs.csv', delimiter=',')    

metrics = val_grouped.columns[2:-1]

results = {}
for metric in metrics:
    t_stat, p_value = ttest_ind(val_grouped[metric], test_combo[metric], equal_var=False)  # Welch's t-test
    u_stat, p_value_u = mannwhitneyu(val_grouped[metric], test_combo[metric], alternative='two-sided')

    results[metric] = {
        "t-test_p": p_value,
        "mannwhitney_p": p_value_u
    }

results_df = pd.DataFrame(results).T
filtered_results = results_df[(results_df["t-test_p"] < 0.05)| (results_df["mannwhitney_p"] < 0.05)]

alpha_bonferroni = 0.05 / 4# len(metrics) # Rec_error + 3*13 features | or 0.05 len(results_df)  
filtered_bonferroni = results_df[
    (results_df["t-test_p"] < alpha_bonferroni) & 
    (results_df["mannwhitney_p"] < alpha_bonferroni)
]
print(filtered_bonferroni)

import matplotlib.pyplot as plt
data_array = [val_grouped['mean_outside'], test_combo['mean_outside'],test_combo['mean_lesion']]
names = ['QUS_in_NASH\n(n={})'.format(len(data_array[0])), 'Oncotech - Liver\n(n={})'.format(len(data_array[1])), 'Oncotech - Lesion\n(n={})'.format(len(data_array[2]))]
boxplot_multi(data_array, names)

#%% Analyzing data from within lesions
import os
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'

os.chdir(workingdir)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

save_path="RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/"
lesions_df = pd.read_csv(save_path+'LesionResults_and_Details_20250325.csv', delimiter=';')

#Plots function
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
def boxplot_multi(data_array, names):
    colorlist = ['#2ca25f','#D7191C','#2C7BB6','#feb24c','#8856a7'] # colors are from http://colorbrewer2.org/
    colors = colorlist[:len(data_array)]
    
    plt.figure()
    i=0
    for don, nom, cor in zip(data_array,names,colors):
        bp = plt.boxplot(don, positions=[i], sym='', widths=0.5)
        set_box_color(bp, cor) 
        i += 1
    
    # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=cor, label=nom)
    plt.legend()
        
    plt.xticks(np.arange(0, len(names)), names, rotation=45)
    plt.tight_layout()
    # plt.show()
    
from scipy.stats import norm
def delong_confidence_interval(y_true, y_scores, alpha=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Get standard error using DeLong’s method
    n1 = sum(y_true == 1)  # Number of positive samples
    n2 = sum(y_true == 0)  # Number of negative samples
    
    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc**2 / (1 + roc_auc)
    
    se = np.sqrt((roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc**2) + (n2 - 1) * (q2 - roc_auc**2)) / (n1 * n2))
    
    # Compute the confidence interval
    z = norm.ppf((1 + alpha) / 2)  # 1.96 for 95% CI
    lower_bound = roc_auc - z * se
    upper_bound = roc_auc + z * se
    
    return roc_auc, (max(0, lower_bound), min(1, upper_bound))

#AUC function
def auc_metrics(data_array, names):
    # Get data
    y_val = np.zeros(len(data_array[0]))
    y_test = np.ones(len(data_array[1]))  
    # Concatenate Rec_error values and labels
    y_true = np.concatenate([y_val, y_test])
    y_scores = np.concatenate([data_array[0], data_array[1]])
    # Compute ROC curve
    fpr, tpr, th = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.fill_between(fpr, tpr, alpha=0.1, color='red')  # Shaded area
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
    # Labels and Title
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve - No lesion vs lesions", fontsize=16)
    # Grid and legend
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=12)
    # plt.show()
    
    youden_index = tpr - fpr
    best_threshold = th[np.argmax(youden_index)]
    y_pred = (y_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # print("Results for %s vs %s" % (names[0], names[1]))
    # print(f"Best Threshold: {best_threshold:.4f}")
    # print(f"Sensitivity (Recall): {sensitivity:.4f}")
    # print(f"Specificity: {specificity:.4f}")
    # print(f"Accuracy: {accuracy:.4f}")
    auc_value, auc_ci = delong_confidence_interval(y_true, y_scores)
    print(f"AUC: {auc_value:.3f}, 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")

    return roc_auc
    
    
def split_ben_mal(df):
    df_ben = df.loc[df['Malign_x']==0]
    df_mal = df.loc[df['Malign_x']==1]
    return df_ben, df_mal
    
metrics = ['mean_lesion', 'std_lesion', 'max_lesion','min_lesion', 'median_lesion', 'skewness_lesion', 'kurtosis_lesion']
# metric = 'mean_lesion'
for metric in metrics:
    # 0 - Everything
    print('Using all datapoints')
    ben, mal = split_ben_mal(lesions_df)
    data_array = [ben[metric], mal[metric]]
    names = ['Oncotech - Benign', 'Oncotech - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    
    # 1 - Selecting Arnaud's or Iman's lesions
    print('Selecting Arnaud and Iman lesions')
    arnaud = lesions_df.loc[lesions_df['Arnaud']==1]
    iman = lesions_df.loc[lesions_df['Iman']==1]
    
    arn_ben, arn_mal = split_ben_mal(arnaud)
    data_array = [arn_ben[metric], arn_mal[metric]]
    names = ['Arnaud - Benign', 'Arnaud - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    ima_ben, ima_mal = split_ben_mal(iman)
    data_array = [ima_ben[metric], ima_mal[metric]]
    names = ['Iman - Benign', 'Iman- Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    # 2 - Selecting by view:
    print('Selecting by view')
    sagittal_df = lesions_df.loc[lesions_df['Details'].str.lower().str.contains('sag')]
    trans_df = lesions_df.loc[lesions_df['Details'].str.lower().str.contains('trans')]
    rest = lesions_df.loc[~lesions_df['Details'].str.lower().str.contains('trans')]
    rest = rest.loc[~rest['Details'].str.lower().str.contains('sag')]
    
    sag_ben, sag_mal = split_ben_mal(sagittal_df)
    data_array = [sag_ben[metric], sag_mal[metric]]
    names = ['Sagittal - Benign', 'Sagittal - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    tra_ben, tra_mal = split_ben_mal(trans_df)
    data_array = [tra_ben[metric], tra_mal[metric]]
    names = ['Transverse - Benign', 'Transverse - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    # 3 - Lesion size
    print('Selecting by lesion size')
    small = lesions_df.loc[lesions_df['Lesion_size']<=lesions_df['Lesion_size'].mean()]
    bigger = lesions_df.loc[lesions_df['Lesion_size']>lesions_df['Lesion_size'].mean()]
    
    sml_ben, sml_mal = split_ben_mal(small)
    data_array = [sml_ben[metric], sml_mal[metric]]
    names = ['Small Lesions - Benign', 'Small Lesions - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    big_ben, big_mal = split_ben_mal(bigger)
    data_array = [big_ben[metric], big_mal[metric]]
    names = ['Bigger Lesions - Benign', 'Bigger Lesions - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    # 4 - Lesion depth
    print('Selecting by lesion depth')
    shallow = lesions_df.loc[lesions_df['Lesion_depth']<=lesions_df['Lesion_depth'].mean()]
    deep = lesions_df.loc[lesions_df['Lesion_depth']>lesions_df['Lesion_depth'].mean()]
    
    sha_ben, sha_mal = split_ben_mal(shallow)
    data_array = [sha_ben[metric], sha_mal[metric]]
    names = ['Shallow Lesions - Benign', 'Shallow Lesions - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    deep_ben, deep_mal = split_ben_mal(deep)
    data_array = [deep_ben[metric], deep_mal[metric]]
    names = ['Deeper Lesions - Benign', 'Deeper Lesions - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    # 5 - CBR B-mode
    print('Selecting by B-mode CNR')
    lowc = lesions_df.loc[lesions_df['CNR_Bmode']<=lesions_df['CNR_Bmode'].mean()]
    highc = lesions_df.loc[lesions_df['CNR_Bmode']>lesions_df['CNR_Bmode'].mean()]
    
    low_ben, low_mal = split_ben_mal(lowc)
    data_array = [low_ben[metric], low_mal[metric]]
    names = ['Lower CNR Lesions - Benign', 'Lower CNR Lesions - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))
    
    high_ben, high_mal = split_ben_mal(highc)
    data_array = [high_ben[metric], high_mal[metric]]
    names = ['Higher CNR Lesions - Benign', 'Higher CNR Lesions - Malignant']
    boxplot_multi(data_array, names)
    auc_val = auc_metrics(data_array, names)
    print(auc_val)
    if auc_val > 0.6: print('CHECK THIS %s - %s \n\n\n' % (metric,names[1]))

#%%# Cooking
'''


import matplotlib.pyplot as plt
import numpy as np

# metrics_chosen = [i for i in metrics_og if 'zero_crossings' not in i]
# metrics_chosen = metrics_chosen[:5]
# data_a = np.array(val_grouped[metrics_chosen])
# data_b = np.array(test_grouped[metrics_chosen])
# ticks = metrics_chosen

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


data_a = data_0
data_b = data_1


plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(np.arange(data_a.shape[1]))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(np.arange(data_b.shape[1]))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Validation - QUS in NASH')
plt.plot([], c='#2C7BB6', label='Test - Oncotech')
plt.legend()

plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, rotation=45)
# plt.xlim(-2, len(ticks)*2)
# plt.ylim(-0.01, 0.01)
plt.tight_layout()
plt.show()

#1
in_signal = np.array(signal)
out_signal = np.array(reconstructed)

sample_no = 0 #Patient number
frame_no = 0 #Frame number
n_frames = 3
n_lines = 254

def group_by_file(signal, reconstructed, sample_no, frame_no, n_frames=3, n_lines=254):
    

    in_signal = np.array(signal)
    out_signal = np.array(reconstructed)
    diff_signal = np.abs(out_signal - in_signal)
 

    # for sig,title in zip([in_signal, out_signal, diff_signal], ['In','Out','Diff']):
    for sig,title in zip([diff_signal], ['Diff']):
        
        repar = sig/sig.max()
        repar = repar*255
        sample_npoints = np.arange(sample_no*n_frames*n_lines,(sample_no+1)*n_frames*n_lines)
        sample_patient = repar[sample_npoints] # 600 signals in dim (1,3248) (if full signal)
        frame_npoints = np.arange(frame_no*n_lines, (frame_no+1)*n_lines)
        frame_RF = sample_patient[frame_npoints]
        # RF_frame = np.transpose(frame_RF)
        # plt.imshow(RF_frame, cmap='gray', aspect='auto'), plt.title('RF'), plt.show()
        
        center = frame_RF[100:150,:]
        left = frame_RF[:100,:]
        right = frame_RF[150:,:]
        
        fig,ax = plt.subplots()
        ax.plot(np.mean(center, axis=0), 'green', label='center')
        ax.plot(np.mean(right, axis=0), 'orange', label='right')
        ax.plot(np.mean(left, axis=0), 'red', label='left')
        ax.plot(np.mean(frame_RF, axis=0), 'black', label='full')
        ax.legend()
        plt.show()
    return diff_signal

diff0 = group_by_file(in_signal, out_signal, sample_no, frame_no)


#2
in_signal = np.array(signal)/np.max(signal)
out_signal = np.array(reconstructed)/np.max(reconstructed)
diff_signal = np.abs(out_signal - in_signal)

aaa = np.mean(diff_signal, axis=0)
plt.plot(aaa), plt.show()

bbb = np.subtract(diff_signal, aaa)
c = np.mean(bbb, axis=0)
plt.plot(c), plt.show()



# Import the lesions classes
lesion_classes = pd.read_csv(lesions_class, delimiter=';')
lesion_classes_name = lesion_classes['patientID'].str.replace("ONCO_TECH", "ONCOTECH", case=False, regex=True)
lesion_classes_name = lesion_classes_name.str.replace("ONCOTECH_", "ONCOTECH", case=False, regex=True)
a = lesion_classes_name.str.upper().str.split('ONCOTECH', expand=True)[1]
lesion_classes['Patient'] = a
final_oncotech = pd.merge(lesion_classes, test_df)




# AUC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get data
y_val = np.zeros(len(val_grouped))  # Class 0 (Validation)
y_test = np.ones(len(test_grouped))  # Class 1 (Test)

# Concatenate Rec_error values and labels
y_true = np.concatenate([y_val, y_test])
y_scores = np.concatenate([val_grouped['Rec_error'], test_grouped['Rec_error']])

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f"AUC = {roc_auc:.3f}")
plt.fill_between(fpr, tpr, alpha=0.1, color='red')  # Shaded area

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)

# Labels and Title
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curve - No lesion vs lesions", fontsize=16)

# Grid and legend
plt.grid(alpha=0.3)
plt.legend(loc="lower right", fontsize=12)

# Show plot
plt.show()






## Checking all patients in Oncotech but ordering it without bs
patids = []

# for i in set(pd.unique(test_df['Patient'])):
for i in set(pd.unique(test_grouped.index)):
    if type(i) == str:
        if '_' in i: 
            i = i.split('_')[1]
    patids.append(int(i))

patids = set(patids)



# Getting the same in/out lesion metrics for validation sets:

import pandas as pd
import torch
import scipy
import matplotlib.pyplot as plt
#load prepare_loader and model functions
valreslist = []

for seed in [9, 19, 21, 25, 42]:
    print('Using seed:', seed)
    path_to_sets = "RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/"
    val = torch.load(path_to_sets + 'val_{}_seed-{}_fullsignal_alllines.pth'.format(data_mode, int(seed)), weights_only=False)
    model = CardioVAE(image_input_channels=1, ecg_input_dim=val[0].shape[-1],latent_dim=n_latents).to(device)
    model.load_state_dict(torch.load(path_to_sets+'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed), weights_only=True))
    print('Model loaded:', 'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed))
    
    valloader,_         = prepare_loaders(val, batch_size=1, shuffle=False)
    print('Dataloader ready.')

    val_pats = pd.Series(val[2]).str.split(r'visit\\',expand=True)[1]
    val_pats = val_pats.str.split(r'QUS_migr', expand=True)[0]
    val_pats = pd.unique(val_pats)
    acquisitions_df = pd.DataFrame()
    acquisitions_df['Acquisition'] = val_pats
    acquisitions_df = acquisitions_df.loc[np.repeat(acquisitions_df.index, 3)].reset_index(drop=True)

    
    signal, reconstructed, labels, val_error = reconstruction_val(model, valloader)


    diff_sig = np.array(signal) # np.abs(np.array(reconstructed)-np.array(signal))
    n_lines = 254
    n_frames = 3
    img_array_shape = [diff_sig.shape[0]//n_lines, n_lines, diff_sig.shape[1]]
    diff_imgs = diff_sig.reshape(img_array_shape).transpose(0, 2, 1)

    out_feats = extract_features2D(diff_imgs, 'outside')
    
    features_list = [out_feats]    
    dataset = pd.DataFrame({**out_feats})
    
        
    final_results = pd.concat([acquisitions_df, dataset], axis=1)
    final_results['seed'] = [seed]*len(final_results)
    final_results.to_csv(path_to_sets+'FeaturesResults/'+'NASH_NoLesion_Input_Comparison_{}.csv'.format(seed))



# Plotting rec_error by file
import matplotlib.pyplot as plt
import numpy as np
# Compute the average Rec_error
avg_rec_error = final_oncotech['Rec_error'].mean()
# Identify Filepaths with Rec_error above average
above_avg_files = final_oncotech.loc[final_oncotech['Rec_error'] > avg_rec_error, 'Filepath'].tolist()
# Define colors based on Malign
colors = final_oncotech['Malign'].map({0: 'green', 1: 'red'}).fillna('gray').tolist()
# Create the figure
plt.figure(figsize=(12, 6))
# Scatter plot of Rec_error with colors based on Malign
plt.scatter(final_oncotech['Filepath'], final_oncotech['Rec_error'], c=colors, alpha=0.7)
# Draw the average line
plt.axhline(avg_rec_error, color='blue', linestyle='dashed', linewidth=2, label=f'Avg Rec_error: {avg_rec_error:.2f}')
# Improve readability of x-ticks
plt.xticks(rotation=90, fontsize=8)
plt.xlabel("Filepath")
plt.ylabel("Rec_error")
plt.title("Rec_error per Filepath with Malign Classification")
plt.legend()
# Show the plot
plt.show()
# Return the list of Filepaths above average
above_avg_files


# Correlation between rec_error and lesion size
import scipy.stats as stats
pearson_corr, p_value = stats.pearsonr(final_lesions['Lesion_size'], final_lesions['Rec_error'])
print(f"Pearson correlation: {pearson_corr:.3f}, p-value: {p_value:.3f}")
spearman_corr, p_value = stats.spearmanr(final_lesions['Lesion_size'], final_lesions['Rec_error'])
print(f"Spearman correlation: {spearman_corr:.3f}, p-value: {p_value:.3f}")
kendall_corr, p_value = stats.kendalltau(final_lesions['Lesion_size'], final_lesions['Rec_error'])
print(f"Kendall correlation: {kendall_corr:.3f}, p-value: {p_value:.3f}")
import seaborn as sns
import matplotlib.pyplot as plt
spearman_corr, p_value = stats.spearmanr(final_lesions['Lesion_size'], final_lesions['Rec_error'])
sns.regplot(x=final_lesions['Lesion_size'], y=final_lesions['Rec_error'], scatter_kws={'s':10}, line_kws={"color":"red"})
# plt.title(f"Pearson correlation: {pearson_corr:.3f}")
plt.title(f"Spearman correlation: {spearman_corr:.3f} (p:{p_value:.3f})")
plt.xlabel("Lesion Size")
plt.ylabel("Rec Error")
plt.show()


###########
# Loading the data for a signal with lesion and measuring the reconstruction errror
import pandas as pd
import numpy as np
from skimage.transform import resize
import os
import torch
workingdir = '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code'
os.chdir(workingdir)
from scipy.io import loadmat
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
def acquire_mask(filename):
    mask_file = filename
    fid = open(mask_file, 'r')
    size_Mask = np.fromfile(fid, np.int32,4)
    mask = np.fromfile(fid, np.int8,-1)
    mask = mask[:30*254*3249] #to avoid one patient with 100 frames
    mask2 = mask.reshape(30,254,3249).T
    return mask2
runcell('Model and Training functions', '//chum.rtss.qc.ca/users/Settings/Folders/Desktop/p0112011/Desktop/Coding/RF_in_NASH_code/1D_VAE_MASHtoOnco.py')
# Define configuration
oncotech_filename = 'Oncotech_Dataset_20250217.csv'
path_to_sets = "RF_dataset/Saved_seeds/VAE_Oncotechonly/"
lesions_class = "Oncotech_Lesion_Class_20250312.csv"
path_to_weights = 'RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/'
biomarker = 'NAS_class'
data_mode = 'RF'

seed = 19
oncotech_file = pd.read_csv(oncotech_filename, delimiter=';') #Open csv as DataFrame
    

all_files = oncotech_file['Filepath']
all_masks = oncotech_file['Maskpath']
assert len(all_files) == len(all_masks)


model = CardioVAE(image_input_channels=1, ecg_input_dim=3248,latent_dim=256).to(device)
model.load_state_dict(torch.load(path_to_weights+'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed), weights_only=True))
print('Model loaded:', 'trained_model_{}/trained_model_{}_{}_seed-{}_bestloss.pth'.format(seed, biomarker, data_mode, seed))
model.to(device)

assert len(all_files) == len(all_masks)

sizes=[]
error=[]

for i in range(len(oncotech_file)):
    # Printing progress
    if i % 5 == 0:
        print('Loading file', i, 'of', len(oncotech_file))

    filename = all_files[i]
    RF_data = loadmat(filename)['RFmig']
    mask = acquire_mask(all_masks[i])

    j = 15 #only using one frame

    single_RF = RF_data[:-1,:,j]
    norm_RF = (single_RF - np.mean(single_RF)) / np.std(single_RF)        # mean 0 + std 1
    single_mask = mask[:-1,:,j] #removing the last line of the 2D image
    
    for k in range(norm_RF.shape[1]):
        single_line = norm_RF[...,k]
        single_limask = single_mask[...,k]
        lesion_size = np.count_nonzero(single_limask)
        
        if lesion_size >= 64: 
            continue

        sizes.append(lesion_size)
        
        with torch.no_grad():
            inputs = np.expand_dims(np.expand_dims(single_line, axis=0),axis=0)
            inputs = torch.Tensor(inputs).to(device)
                    
            _, val_recon_ecg_only, val_mu_ecg, val_logvar_ecg = model(ecg=inputs)
            val_ecg_loss, mse = elbo_loss(None, None, val_recon_ecg_only, inputs, val_mu_ecg, val_logvar_ecg, 0, 10, 1)
                
        error.append(val_ecg_loss.item())
import seaborn as sns
import scipy.stats as stats
spearman_corr, p_value = stats.spearmanr(sizes, error)
sns.regplot(x=sizes, y=error, scatter_kws={'s':10}, line_kws={"color":"red"})
plt.title(f"Spearman correlation: {spearman_corr:.3f} (p:{p_value:.3f})")
plt.xlabel("Lesion Size")
plt.ylabel("Rec Error")
plt.show()        

# Saving list to file
with open(path_to_weights+"RecError_LinesWithoutLesion.txt", "w") as f:
    for s in error:
        f.write(str(s) +"\n")



# Reading txt files and comparing them

import numpy as np
from scipy.stats import ttest_ind

# Read numbers from two text files
def read_txt_file(filename):
    with open(filename, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

# Load data from both files
data1 = read_txt_file('RecError_LinesWithoutLesion.txt')
data2 = read_txt_file('RecError_LinesWithLesion.txt')

# Run an independent t-test
t_stat, p_value = ttest_ind(data1, data2, equal_var=False)  # Welch's t-test

# Print results
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Significant difference between the two groups (reject H0).")
else:
    print("No significant difference (fail to reject H0).")

plt.figure()
## df = pd.read_csv('FeaturesResults/MultiFeatures_CARDIOVAE_NAS_class_RF_seed-19-val.csv')
## data3 = df['Rec_error']
bpl = plt.boxplot(data3, positions=[0], sym='', widths=0.6) #data3 came from MultiFeatures_CARDIOVAE_NAS_class_RF_seed-19-val.csv'
bpr = plt.boxplot(data1, positions=[1], sym='', widths=0.6)
bps = plt.boxplot(data2, positions=[2], sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')
set_box_color(bps, 'green')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Validation - QUS in NASH')
plt.plot([], c='#2C7BB6', label='Test - Oncotech No Lesion')
plt.plot([], c='green', label='Test - Oncotech Lesion')
plt.legend()

# ticks = ['QUS in NASH', 'Oncotech No']
# plt.xticks(np.arange(0, len(ticks)), ticks, rotation=45)
# plt.xlim(-2, len(ticks)*2)
# plt.ylim(-0.01, 0.01)
plt.tight_layout()
plt.show()







# Checking the similarity between sets
## From https://github.com/veflo/uncert_quant // https://arxiv.org/html/2405.01978v1
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
def quantify_similarity(datasets_to_compare, x_mesh):
    """
    Quantify the similarity between the training dataset and other datasets using KL divergence and JS distance.

    Args:
    - datasets_to_compare (dict): A dictionary containing datasets to compare. 
                                  Keys represent dataset names and values represent the datasets.
    - x_mesh (ndarray): Meshgrid for evaluating PDFs.

    Returns:
    - kl_divergences (dict): Dictionary containing KL divergences for each dataset compared to the training dataset.
    - js_dist (dict): Dictionary containing JS distances for each dataset compared to the training dataset.
    """

    # Store KDE estimator for the training dataset
    kde_data_train = gaussian_kde(datasets_to_compare['data_train'].T)
    x = x_mesh
    eps = np.finfo(float).eps
    kde_data_train_pdf = kde_data_train.pdf(x.T)

    # Compute KL and JS divergences between the training dataset and other datasets
    kl_divergences = {}
    js_dist = {}
    for name, data in datasets_to_compare.items():
        if name != 'data_train':
            kde_data = gaussian_kde(data.T)
            kde_data_pdf = kde_data.pdf(x.T) + eps
            kl_div = entropy(kde_data_train_pdf, kde_data_pdf)
            js = jensenshannon(kde_data_train_pdf, kde_data_pdf)
            kl_divergences[name] = kl_div
            js_dist[name] = js
            print(f"Completed calculations for dataset: {name}")

    # Print KL and JS dist in a tabular form
    print() # Line shift 
    print("KL-div and JS Distance between the training dataset and other datasets:")
    print("{:<20} {:<20} {:<20}".format("Dataset", "KL-Div", "JS Distance"))
    for name, kl_div in kl_divergences.items():
        js = js_dist[name]
        print("{:<20} {:.2f} {:>18.2f}".format(name, kl_div, js))

    return kl_divergences, js_dist
import numpy as np
# Assume data_train and data_test are (1000, 1, 3248)
data_train_flat = train[0].flatten()
data_val_flat = val[0].flatten()
data_test_flat = test[0].flatten()

data_train_kde = data_train_flat[np.newaxis, :]
data_val_kde = data_val_flat[np.newaxis, :]
data_test_kde = data_test_flat[np.newaxis, :]

x_min = min(data_train_flat.min(), data_test_flat.min())
x_max = max(data_train_flat.max(), data_test_flat.max())
x_mesh = np.linspace(x_min, x_max, 1000).reshape(-1, 1)

datasets_to_compare = {
    'data_train': data_train_kde.T,
    'data_val': data_val_kde.T,
    'data_test': data_test_kde.T
}

kl_divs, js_dists = quantify_similarity(datasets_to_compare, x_mesh)
'''






































#%%
## Trying localization
# First run the reconstruction_val function
# # signal, reconstructed, labels, val_error = reconstruction_val(model, testloader)
from utils.loading_data import rf2bmode as r2b
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
import matplotlib.patches as pac

# Getting the frames and scores
n_frames = 3
n_lines = 254
frames = []
scores = []
reconstructions = []

for sample_no in range(0,int(len(signal)/n_frames/n_lines)):
    sample_npoints = np.arange(sample_no*n_frames*n_lines,(sample_no+1)*n_frames*n_lines)
    frame0_points = sample_npoints[0:nlines]
    frame1_points = sample_npoints[nlines:nlines*2]
    frame2_points = sample_npoints[nlines*2:nlines*3]
    
    for frame in [frame0_points, frame1_points, frame2_points]:
        sample = np.array(signal)[frame] # signals in dim (254,3248) (if full signal)
        recs = np.array(reconstructed)[frame]
        RF_frame = np.transpose(sample)
        errors = np.array(val_error)[frame]
        
        frames.append(RF_frame)
        scores.append(errors)
        reconstructions.append(recs)

# Getting all masks
from skimage import measure
def acquire_mask(filemask):
    fid = open(filemask, 'r')
    mask = np.fromfile(fid, np.int8,-1)
    mask = mask[:30*254*3249] #to avoid one patient with 100 frames
    mask2 = mask.reshape(30,254,3249).T
    return mask2

def peak_idxs(line, thresh=0.7511*3248):    
    peak_idx = np.argmax(line)
    peak_value = line[peak_idx]
    threshold = 0.80 * peak_value
    
    # Find left bound
    left_idx = peak_idx
    while left_idx > 0 and line[left_idx] > threshold:
        left_idx -= 1
    
    # Find right bound
    right_idx = peak_idx
    while right_idx < len(line) - 1 and line[right_idx] > threshold:
        right_idx += 1
    
    indices_around_peak = np.arange(left_idx, right_idx + 1)
    
    # multiple_peaks = np.where(line > 0.50*np.max(line))[0] 
    multiple_peaks = np.where(line > thresh)[0] 
    
    return peak_idx, indices_around_peak, multiple_peaks

oncotech_filename = 'Oncotech_Dataset_20250217.csv'
oncotech_file = pd.read_csv(oncotech_filename, delimiter=';') #Open csv as DataFrame
list_of_masks = oncotech_file['Maskpath']
contours = []
for filemask in list_of_masks:
    mask = acquire_mask(filemask)
    for j in range(1,30,10):
        single_mask = mask[...,j]
        mask_contours = measure.find_contours(single_mask, level=0.5)
        contours.append(mask_contours)

#%% Getting distance metrics
path_to_imgs = 'RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/DetectionLocalizationImgs/'
chosenids = range(0,len(frames),3)
coords = []
overlap_on_x = 0
x_on_max = 0
x_on_peaks = 0
overlap_on_y = 0
y_on_max = 0
y_on_peaks = 0
below_thresh = 0

for idx, chosenid in enumerate(chosenids):
    flag_atMax = 0
    flag_atpeaks = 0
    flag_atMaxY = 0
    flag_atpeaksY = 0

    img = r2b(frames[chosenid])
    msk = contours[chosenid]
    if len(msk)>1: print(chosenid, 'has more than one contour'); msk=[msk[0]]
    sco = scores[chosenid]
    rec = reconstructions[chosenid]
    
    
    for contour in msk:
        boxdims = [np.min(contour[:,1]), np.max(contour[:,1]), np.min(contour[:,0]), np.max(contour[:,0])]
        x0 = boxdims[0]
        y0 = boxdims [2]
        width = boxdims[1]-boxdims[0]
        height = boxdims[3]-boxdims[2]
    
        centroid_y = np.mean(contour[:, 0])
        centroid_x = np.mean(contour[:, 1])

    gaussline = gaussian_filter1d(sco, 10)

    ex_RF = frames[chosenid]
    ex_Rec = np.transpose(rec)
    peak, rangePeak, allPeaks = peak_idxs(gaussline,thresh=0.7511*3248)
    
    grouped_rfs = ex_RF[:,rangePeak]/np.max(ex_RF)
    grouped_recs = ex_Rec[:,rangePeak]/np.max(ex_Rec)
    
    avg_both = np.cumsum((grouped_recs - grouped_rfs),axis=1)
    a = avg_both[:,-1]
    gaussdiff = gaussian_filter1d(a, 50)
    
    ypeak, yrangePeak, yallPeaks = peak_idxs(gaussdiff, thresh=0.5*np.max(gaussdiff))
    det_x0 = rangePeak[0]
    det_width = rangePeak[-1]-det_x0
    det_y0 = yrangePeak[0]
    det_height = yrangePeak[-1]-det_y0  

    if int(centroid_x) in rangePeak:
        x_on_max += 1 
        flag_atMax = 1

    if len(allPeaks) > 1:
        if int(centroid_x) in allPeaks:
            x_on_peaks += 1
            flag_atpeaks = 1
            
        if x0 <= allPeaks[-1] and x0 >= allPeaks[0]:
            overlap_on_x += 1
    if np.max(gaussline) < 0.7511*3248:
        below_thresh += 1 
        print(chosenid, 'has maximum rec_error below thresh');

    if int(centroid_y) in yrangePeak:
        y_on_max += 1 
        flag_atMaxY = 1

    if len(allPeaks) > 1:
        if int(centroid_y) in yallPeaks:
            y_on_peaks += 1
            flag_atpeaksY = 1
            
        if y0 <= yallPeaks[-1] and y0 >= yallPeaks[0]:
            overlap_on_y += 1
        
    entry = {
        'chosenId': chosenid,
        'file': oncotech_file.loc[idx]['Filepath'],
        'rec_error': np.mean(sco),
        'mask_x': centroid_x,
        'mask_y': centroid_y,
        'mask_box': [x0, y0, width, height],
        'det_x': peak,
        'det_y': ypeak,
        'det_box': [det_x0, det_y0, det_width, det_height],
        'loc_Peak': flag_atMax,
        'loc_other': flag_atpeaks,
        'loc_PeakY': flag_atMaxY,
        'loc_otherY': flag_atpeaksY,
        
        }
    coords.append(entry)
    
    

df = pd.DataFrame(coords)
df.to_csv(path_to_imgs+'coords_data3.csv')


#%% Plotting
path_to_imgs = 'RF_dataset/Saved_seeds/VAE_NASHonly/oldSplit/FeaturesResults/DetectionLocalizationImgs/'
import cv2
def contour_area(contour):
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def largest_contour(countours):
    areas = [contour_area(c) for c in countours]
    largest_idx = np.argmax(areas)
    largest_contour = countours[largest_idx]
    return largest_contour

def find_smallest_distance(contours, centroid):
    best_x = np.inf
    best_y = np.inf
    centroid_x = centroid[0]
    centroid_y = centroid[1]
    
    for c in contours:
        centroid_y_pred = np.mean(c[:, 0])
        centroid_x_pred = np.mean(c[:, 1])
        
        seg_distance_x = np.abs(centroid_x_pred - centroid_x)
        seg_distance_y = np.abs(centroid_y_pred - centroid_y)

        best_distance = ((best_x/254)**2)+((best_y/3248)**2)
        new_distance = ((seg_distance_x/254)**2)+((seg_distance_y/3248)**2)
        # print('Previous best distance: ', best_distance)
        # print('New distance: ',  new_distance)

        if new_distance < best_distance:
            best_x = seg_distance_x
            best_y = seg_distance_y
            best_cont = c
            best_centroid = [centroid_x_pred, centroid_y_pred]
    
    return best_cont, [best_x, best_y], best_centroid

# Good results: 294, 288, 717

# chosenid = 719 #example
# chosenids = [50, 81, 200, 218, 601, 666] #random numbers
# chosenids = [288, 285, 450, 510, 447] # top errors
# chosenids = [66, 96, 261, 288, 294, 468, 489, 495, 510, 711, 714, 717] # Good results
# chosenids = [513, 3, 99, 612, 126] #Bad results
# chosenids = [63,108,528,543,612] # more than one mask
# chosenids = [6, 138, 141, 573, 636] # below threshold

chosenids = range(0,len(frames),3) #all
coords = []
distances_in_x = []
distances_in_y = []


for idx, chosenid in enumerate(chosenids):
    drange = [35,100]
    img = r2b(frames[chosenid])
    msk = contours[chosenid]
    if len(msk)>1: print(chosenid); msk=[msk[0]]
    sco = scores[chosenid]
    rec = reconstructions[chosenid]
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.30], width_ratios=[1,0.30])  # Change width ratio
    # gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.30])
    
    # Frame image
    vmin, vmax = np.percentile(img, drange)  # or choose any suitable range
    axs0 = plt.subplot(gs[0])
    axs0.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    for contour in msk:
        boxdims = [np.min(contour[:,1]), np.max(contour[:,1]), np.min(contour[:,0]), np.max(contour[:,0])]
        x0 = boxdims[0]
        y0 = boxdims [2]
        width = boxdims[1]-boxdims[0]
        height = boxdims[3]-boxdims[2]
    
        centroid_y = np.mean(contour[:, 0])
        centroid_x = np.mean(contour[:, 1])
        axs0.plot(contour[:, 1], contour[:, 0], linewidth=1, linestyle='dashed', color='orange')  # X = column, Y = row
    axs0.scatter(centroid_x, centroid_y, color='red', marker='x', label='Centroid')
    centroid = [centroid_x, centroid_y]
    axs0.set_aspect('auto')
    axs0.set_yticks(np.linspace(0, img.shape[0], 5))
    axs0.set_yticklabels(np.linspace(0, 240, 5, dtype=int), fontsize=14)
    axs0.set_ylabel("Depth (mm)", fontsize=18)
    axs0.set_xticks([])
    axs0.set_title(chosenid)
    rectangle = pac.Rectangle((x0,y0), width, height, fc = 'none', ec="red", linestyle='--')
    axs0.add_patch(rectangle)
    
    
    # Errors plot
    axs1 = plt.subplot(gs[2])
    gaussline = gaussian_filter1d(sco, 10)
    axs1.plot(gaussline)
    axs1.axhline(y=0.75*3248, ls='--', dashes=(5,5), lw=1.5)
    
    axs1.set_yticks([])
    axs1.set_xticks([img.shape[1] // 2])
    axs1.set_xticklabels(["0"], fontsize=14)
    axs1.set_xlabel(r"$\theta$ (rad)", fontsize=18)
    axs1.set_ylabel("Anomaly \nscore", fontsize=14)
    axs1.set_xlim([0,len(sco)])
    if np.max(sco) <= 0.75*3248:
        axs1.set_ylim([np.min(sco),5000])
    else:
        axs1.set_ylim([np.min(sco),np.max(gaussline)*2])
    axs1.axvline(x=x0, color='red', ls='--', dashes=(5,5), lw=2)  
    axs1.axvline(x=boxdims[1], color='red', ls='--', dashes=(5,5), lw=2)  
    
    
    # # Depth plot
    axs2 = plt.subplot(gs[1])
    ex_RF = frames[chosenid]
    ex_Rec = np.transpose(rec)
    peak, rangePeak, allPeaks = peak_idxs(gaussline)
    
    grouped_rfs = ex_RF[:,rangePeak]/np.max(ex_RF)
    grouped_recs = ex_Rec[:,rangePeak]/np.max(ex_Rec)
    

    avg_rfs = np.mean(grouped_rfs,axis=1)
    avg_recs = np.mean(grouped_recs,axis=1)
    avg_both = np.cumsum((grouped_recs - grouped_rfs),axis=1)
    a = avg_both[:,-1]
    gaussdiff = gaussian_filter1d(a, 50)
    
    
    
    axs2.plot(gaussdiff, range(len(gaussdiff)))
    axs2.invert_yaxis()
    axs2.set_yticks([])
    axs2.set_ylim([len(gaussdiff),0])
    axs2.set_xticks([])
    axs2.axhline(y=y0, color='red', ls='--', dashes=(5,5), lw=1.5)  
    axs2.axhline(y=boxdims[3], color='red', ls='--', dashes=(5,5), lw=1.5)  
    
    #Plot segmentation mask
    axs3 = plt.subplot(gs[3])
    gain = gaussline.copy()
    difference_map = np.abs(np.transpose(rec) - frames[chosenid])
    mesh = gain * difference_map
    map_threshold = np.mean(mesh)*1.5
    binary_image = (mesh > map_threshold).astype(np.uint8)
    # smoothed = gaussian_filter(binary_image, sigma=0.2)
    blurred = cv2.GaussianBlur(binary_image, (21, 21), 0)  # (5,5) is the kernel size
    axs3.imshow(blurred , cmap='jet', aspect='auto')
    axs3.set_yticks([])
    axs3.set_xticks([])
    # axs3.axis('off')
    
    blurred_contours = measure.find_contours(blurred, level=0.5)

    ## If we want to use only the largest contours:
    blurred_contours = [largest_contour(blurred_contours)]
    ## Else, we will find the closest:
    largest_cont, seg_distances, pred_centroids = find_smallest_distance(blurred_contours, centroid)

    distances_in_x.append(seg_distances[0])
    distances_in_y.append(seg_distances[1])    

    axs0.plot(largest_cont[:, 1], largest_cont[:, 0], linewidth=1, linestyle='dashed', color='green')  # X = column, Y = row
    axs0.scatter(pred_centroids[0], pred_centroids[1], color='green', marker='x', label='Centroid')
    axs2.set_title('Difference\n signal', fontsize=14)
    axs3.set_xlabel('Localization', fontsize=14)
    
    plt.tight_layout()
    # axs0.scatter(peak, np.argmax(gaussdiff), color='green', marker='o', label='Localization')
    
    ypeak, yrangePeak, yallPeaks = peak_idxs(gaussdiff)
    det_x0 = rangePeak[0]
    det_width = rangePeak[-1]-det_x0
    det_y0 = yrangePeak[0]
    det_height = yrangePeak[-1]-det_y0
    
    # det_rectangle = pac.Rectangle((det_x0,det_y0), det_width, det_height, fc = 'none', ec="green", linestyle='--')
    # axs0.add_patch(det_rectangle)
    axs1.text(0,0.77*3248,'Detection \nthreshold',fontsize=8, color='#1f77b4')
    
    # name = 'Localization_'+str(chosenid)+'_withLargestContour.png'
    # plt.savefig(path_to_imgs+name)
    
    entry = {
        'chosenId': chosenid,
        'file': oncotech_file.loc[idx]['Filepath'],
        'rec_error': np.mean(sco),
        'mask_x': centroid_x,
        'mask_y': centroid_y,
        'mask_box': [x0, y0, width, height],
        'det_x': peak,
        'det_y': ypeak,
        'det_box': [det_x0, det_y0, det_width, det_height],
        'loc_centr_x': pred_centroids[0],
        'loc_centr_y': pred_centroids[1]
        }
    coords.append(entry)
    
    plt.show()    
    
df = pd.DataFrame(coords)
df.to_csv(path_to_imgs+'coords_data_LargestLoc.csv')    


#%%

print('Average distances are- X: %.2f +- %.2f; Y: %.2f +- %.2f' % 
      (np.mean(distances_in_x), np.std(distances_in_x), np.mean(distances_in_y), np.std(distances_in_y)))
