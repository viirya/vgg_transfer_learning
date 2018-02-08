
import os

from numpy import array, random

def get_files_in_dir(dirname, random_flag = False):

    files = []
    fullpath_files = []
    filelist = os.listdir(dirname)
    for filename in filelist:

        full_filename = dirname + '/' + filename
        files.append(filename)
        fullpath_files.append(full_filename)

    if random_flag == True:
        random.shuffle(files)
        random.shuffle(fullpath_files)

    return (array(files), array(fullpath_files))

def load_file(filename):

    f = open(filename, 'r')
    content = []
    for line in f:
        content.append(line.rstrip())

    f.close()

    return array(content)

def write_file(data, filename):

    f = open(filename, 'w')

    for item in data:
        f.write(item + "\n")

    f.close()
