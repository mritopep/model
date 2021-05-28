import os

PWD = str(os.path.dirname(os.path.abspath(__file__)))
DATA = PWD.replace("/model", "/data")
SCRIPT = PWD.replace("/model", "/scripts")
TOKEN = PWD.replace("/model", "/tokens")


# Locations
DOWNLOAD = f'{DATA}/downloads'
ADNI = f'{DATA}/dataset/adni'
CANCER = f'{DATA}/dataset/cancer'
EXTRACT = f'{DATA}/extracts'
METADATA_ADNI = f'{DATA}/metadata/adni'
METADATA_CANCER = f'{DATA}/metadata/cancer'
DIVIDE = f'{DATA}/divided_data'
PREPROCESSED = f'{DATA}/preprocessed_data'
FILTERED = f'{DATA}/filtered_data'
ZIPPED = f'{DATA}/zip_data'
PICKLE = f'{DATA}/pickle'
POSTPROCESS = f'{DATA}/postprocess'

# Temp
SKULL_STRIP = f'{DATA}/temp/skull_strip'
IMG_REG = f'{DATA}/temp/img_reg'
DENOISE = f'{DATA}/temp/denoise'
PETPVC = f'{DATA}/temp/petpvc'
BAIS_COR = f'{DATA}/temp/bias_cor'
TEMP_OUTPUT = f'{DATA}/temp/output'

# Postprocess Temp
IMG1_MRI = f'{DATA}/temp/img1/mri'
IMG1_PET = f'{DATA}/temp/img1/pet'
IMG2_MRI = f'{DATA}/temp/img2/mri'
IMG2_PET = f'{DATA}/temp/img2/pet'

# Shell Scripts
SHELL = f"{SCRIPT}/shell_scripts"

# Full paths
DATA_PATHS = [DOWNLOAD, ADNI, CANCER, EXTRACT, METADATA_ADNI,
              METADATA_CANCER, DIVIDE, PREPROCESSED, ZIPPED, PICKLE,FILTERED]
TEMP_PATHS = [SKULL_STRIP, IMG_REG, DENOISE, PETPVC, BAIS_COR, TEMP_OUTPUT]
POSTPROCESS_TEMP_PATHS = [IMG1_MRI, IMG1_PET, IMG2_MRI, IMG2_PET]
SCRIPT_PATHS = []
