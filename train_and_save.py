from pix2pix import Pix2Pix
from os import path
import sys
import os
from general import make_dir, make_archive
from gdrive import Gdrive

if __name__ == "__main__":
    tag = ""
    PWD = str(os.getcwd())

    print("Initializing and loading model...")
    gan = Pix2Pix("img_data")
    g = Gdrive()

    print("Starting training ...")


    # e is number of epochs. It should be increased to ~15 to 20 at least.
    e = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    # save_epoch_model if 1 save model of each epoch
    save_epoch_model = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # save_to_drive if 1 upload model to drive
    save_to_drive = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # save_to_drive if 1 upload model to drive
    save_dataset = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    # tag to distungiush
    tag = f"{str(sys.argv[5])}_{e}" if len(sys.argv) > 5 else ""

    if(save_dataset == 1):
        make_archive("img_data", f"{tag}_img_data.zip")
        g.upload(f"{tag}_img_data.zip")

    if(save_epoch_model == 1):
        for epoch in gan.train(epochs=e, batch_size=1, include_val=False, step_print=100):
            folder = os.path.join("saved_epoch_models",
                                  "saved_model_" + str(epoch))
            make_dir([folder])
            print("Saving ..")
            gan.generator.save(os.path.join(
                folder, "gen_model_" + str(epoch) + ".h5"))
            gan.discriminator.save(os.path.join(
                folder, "disc_model_" + str(epoch) + ".h5"))
            gan.combined.save(os.path.join(
                folder, "gan_model_" + str(epoch) + ".h5"))

            if(save_to_drive == 1):
                print("Models Uploading...")
                make_archive(folder, os.path.join(
                    "saved_epoch_models", f"{tag}_part_{str(epoch)}.zip"))
                g.upload(os.path.join("saved_epoch_models",
                         f"{tag}_{str(epoch)}.zip"))
    else:
        value = gan.train(epochs=e, batch_size=1, include_val=False, step_print=100)

    folder = "saved_models"

    print("Training complete. Saving ...")

    gan.generator.save(path.join(folder, "gen_model_" + str(e) + ".h5"))
    gan.discriminator.save(path.join(folder, "disc_model_" + str(e) + ".h5"))
    gan.combined.save(path.join(folder, "gan_model_" + str(e) + ".h5"))

    print("Models saved.")

    print("Models Uploading...")
    PWD = str(os.getcwd())
    make_archive(os.path.join(PWD,"saved_models"),os.path.join(PWD, f"{tag}_saved_models.zip"))
    g.upload(os.path.join(PWD, f"{tag}_saved_models.zip"))
