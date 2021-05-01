folder_a = "/kaggle/input/new-mri-pet/new_data/A/train/"
folder_b = "/kaggle/input/new-mri-pet/new_data/B/train/"
files = sorted(listdir(folder_a))
mr_list, pet_list = [], []
new_shape = (256,256,3)

for i in range(len(files)):
    
    mr_data = cv2.imread(folder_a + files[i])
    pet_data = cv2.imread(folder_b + files[i])
    
    t_mr = np.zeros(new_shape, dtype=np.uint8)
    t_pet = np.zeros(new_shape, dtype=np.uint8)
    
    for i in range(new_shape[-1]):
        t_mr[:,:,i] = pad_2d(mr_data[:,:,i], new_shape[0], new_shape[1])
        t_pet[:,:,i] = pad_2d(pet_data[:,:,i], new_shape[0], new_shape[1])
        
    t_mr = pre_process_mri(t_mr, gamma_correction = True)
    
    mr_list.append(t_mr)
    pet_list.append(t_pet)
    
    
mr_list = np.asarray(mr_list)
pet_list = np.asarray(pet_list)
print(mr_list.shape, pet_list.shape)
dataset = [mr_list, pet_list]
print("Input :",dataset[0].shape, "Output :",dataset[1].shape)

from numpy import savez_compressed

filename = 'gamma_corrected_test_data.npz'
savez_compressed(filename, mr_list, pet_list)