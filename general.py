import os
import pickle
import gzip
import shutil
from zipfile import ZipFile
from paths import *


def upzip_gz(input, output):
    with gzip.open(input, 'rb') as f_in:
        with open(output, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def store_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def get_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_dir(dirs):
    for dir in dirs:
        try:
            os.makedirs(dir)
        except:
            pass


def remove_dir(dirs):
    for dir in dirs:
        try:
            os.removedirs(dir)
        except:
            pass


def update_progress(curr_count, total_count):
    progress = int((curr_count/total_count)*100)
    bar = int(progress/2)
    time_taken = total_count - curr_count
    print(f"\r[{'#'*bar}] {progress}% \n ESTIMATED TIME: {time_taken*10} min")


def extract_progress(curr_count, total_count):
    progress = int((curr_count/total_count)*100)
    bar = int(progress/2)
    print(f"\r[{'#'*bar}] {progress}%", end="\r")


def download_progress(curr_byte):
    gb_data = 1024*1024*1024
    progress = int((curr_byte/gb_data)*100)
    bar = int(progress/2)
    print(f"\r[{'#'*bar}] {progress}%", end="\r")


def lock(data, val):
    if val:
        return data[:2]
    else:
        return data


def show_data(tag, data):
    for num, data in enumerate(data):
        print(f"\n{tag}:{num} - {data}\n")


def list_directory(dir):
    try:
        return os.listdir(dir)
    except:
        print("\nNO DIR\n")
        return []


def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)


def divide(name, path, parts):
    print("\nDIVIDING FILES\n")
    divided_dirs = []
    dirs = []

    dir_list = os.listdir(path)
    num_dir = len(dir_list)
    div = int(num_dir/parts)

    for i in range(1, num_dir+1):
        if(i % div == 0):
            dirs.append(dir_list[i-1])
            divided_dirs.append(dirs)
            dirs = []
        else:
            dirs.append(dir_list[i-1])

    if(dirs not in divided_dirs and len(dirs) != 0):
        divided_dirs.append(dirs)

    dir_count = 0
    file_count = 0
    for divide in divided_dirs:
        dir_count += 1
        for folder in divide:
            file_count += 1

    print(f"\nDIR COUNT:{dir_count} FILE COUNT: {file_count}\n")

    count = 0

    for divide in divided_dirs:
        os.makedirs(f"{path}/{name}")
        divided_file_paths = []
        count += 1
        for folder in divide:
            dest_directory = f"{path}/{name}/{folder}"
            src_directory = f"{path}/{folder}"
            os.makedirs(dest_directory)
            shutil.move(f"{src_directory}/mri.nii",
                        f"{dest_directory}/mri.nii")
            shutil.move(f"{src_directory}/pet.nii",
                        f"{dest_directory}/pet.nii")
            shutil.rmtree(src_directory)
        update_progress(count, file_count)
