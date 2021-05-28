from pix2pix import Pix2Pix
from os import path

if __name__ == "__main__":
    print("Initializing and loading model...")
    gan = Pix2Pix("img_data")

    print("Starting training ...")

    # e is number of epochs. It should be increased to ~15 to 20 at least. 
    e = 2
    gan.train(epochs=e, batch_size=1, include_val = False, step_print = 100)
    
    folder = "saved_models"
    
    print("Training complete. Saving ...")
    
    gan.generator.save( path.join(folder, "gen_model_" + str(e) + ".h5") )
    gan.discriminator.save( path.join(folder, "disc_model_" + str(e) + ".h5") )
    gan.combined.( path.join(folder, "gan_model_" + str(e) + ".h5") )
    
    print("Models saved.")
