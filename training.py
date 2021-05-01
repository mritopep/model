import time
from util import generate_real_samples, generate_fake_samples


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1, print_step=100):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)

    n_samples = len(trainA)

    n_steps = bat_per_epo * n_epochs
    start_time = time.time()

    d1_sum, d2_sum, g_sum = 0, 0, 0
    epoch_no = 0

    for i in range(n_steps):

        [X_realA, X_realB], y_real = generate_real_samples(
            dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        d1_sum += d_loss1
        d2_sum += d_loss2
        g_sum += g_loss

        if (i+1) % print_step == 0:
            time_taken = time.time() - start_time
            print('step : %d time taken : %.3fs' % (i+1, time_taken))
            start_time = time.time()

        if(i+1) % n_samples == 0:
            epoch_no += 1
            print('Epoch [%d/%d] : Avg. loss => discriminator loss (real imgs) : %.3f | discriminator loss (fake imgs) : %.3f | generator loss : %.3f' % (
                epoch_no, n_epochs, d1_sum/n_samples, d2_sum/n_samples, g_sum/n_samples))

            d1_sum, d2_sum, g_sum = 0, 0, 0

        if (i+1) % (bat_per_epo * 50) == 0:
            filename = '_%06d.h5' % (i+1)
            g_model.save("g_model" + filename)
            d_model.save("d_model" + filename)
            print('>Saved: %s' %
                  ("g_model" + filename + " and " + "d_model" + filename))

    filename = '_ep_%06d.h5' % (n_epochs)
    g_model.save("g_model" + filename)
    d_model.save("d_model" + filename)
    print('>Saved epoch: %s' %
          ("g_model" + filename + " and " + "d_model" + filename))
    print('Model training finished.')


if __name__=="__main__":
    image_shape = dataset[0].shape[1:]
    discriminator = Discriminator(image_shape)
    generator = Generator(image_shape)
    gan_model = P2PGAN(generator, discriminator, image_shape)
    train(discriminator, generator, gan_model, dataset, n_epochs = 35)