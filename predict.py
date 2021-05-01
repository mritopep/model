processed_data = "/content/gamma_corrected_test_data.npz"
dataset = load_real_samples(processed_data)
print("Input :",dataset[0].shape, "Output :",dataset[1].shape)

image_shape = dataset[0].shape[1:]
[X1, X2] = dataset

g_model_file = '/content/g_model_ep_000035.h5'

generator = load_model(g_model_file)

n = 245
m = 50
inp_images, out_images = X1[n : n+m], X2[n:n+m]
fake_images = np.zeros(out_images.shape)

for i in range(len(inp_images)):
    fake_images[i] = generator.predict(inp_images[i:i+1])

proc_imgs = np.copy(fake_images)

for i in range(len(inp_images)):
    proc_imgs[i] = post_process(proc_imgs[i:i+1], thresholding = False)
    
interact(display_image, image_z=(0,proc_imgs.shape[0]-1), npa = fixed(proc_imgs));
interact(display_image, image_z=(0,out_images.shape[0]-1), npa = fixed(out_images));
interact(display_image, image_z=(0,out_images.shape[0]-1), npa = fixed(inp_images))

score, diff_res = calc_avg_ssim(fake_images, out_images)

print("SSIM: {}".format(score))
#interact(display_image, image_z=(0,diff_res.shape[0]-1), npa = fixed(diff_res))
