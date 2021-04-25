from __future__ import print_function
from PIL import Image
import os, sys
import cv2


"""removing general images that are not belong to any class """
def removing_general():
    all_ok_files = ['defense_1_ok.txt', 'defense_2_ok.txt', 'defense_3_ok.txt','defense_4_ok.txt','defense_5_ok.txt',
                 'eiffel_1_ok.txt','eiffel_2_ok.txt','eiffel_3_ok.txt','eiffel_4_ok.txt','eiffel_5_ok.txt',
                 'invalides_1_ok.txt', 'invalides_2_ok.txt','invalides_3_ok.txt','invalides_4_ok.txt','invalides_5_ok.txt',
                 'louvre_1_ok.txt','louvre_2_ok.txt','louvre_3_ok.txt','louvre_4_ok.txt','louvre_5_ok.txt',
                 'moulinrouge_1_ok.txt', 'moulinrouge_2_ok.txt','moulinrouge_3_ok.txt','moulinrouge_4_ok.txt','moulinrouge_5_ok.txt',
                 'museedorsay_1_ok.txt', 'museedorsay_2_ok.txt', 'museedorsay_3_ok.txt', 'museedorsay_4_ok.txt', 'museedorsay_5_ok.txt',
                 'notredame_1_ok.txt', 'notredame_2_ok.txt', 'notredame_3_ok.txt', 'notredame_4_ok.txt', 'notredame_5_ok.txt',
                 'pantheon_1_ok.txt','pantheon_2_ok.txt','pantheon_3_ok.txt','pantheon_4_ok.txt','pantheon_5_ok.txt',
                 'pompidou_1_ok.txt', 'pompidou_2_ok.txt', 'pompidou_3_ok.txt', 'pompidou_4_ok.txt', 'pompidou_5_ok.txt',
                 'sacrecoeur_1_ok.txt','sacrecoeur_2_ok.txt','sacrecoeur_3_ok.txt','sacrecoeur_4_ok.txt','sacrecoeur_5_ok.txt',
                 'triomphe_1_ok.txt','triomphe_2_ok.txt','triomphe_3_ok.txt','triomphe_4_ok.txt','triomphe_5_ok.txt']

    corrupted = ["paris_louvre_000136.jpg","paris_louvre_000146.jpg","paris_moulinrouge_000422.jpg",
                 "paris_museedorsay_001059.jpg","paris_notredame_000188.jpg","paris_pantheon_000284.jpg",
                 "paris_pantheon_000960.jpg","paris_pantheon_000974.jpg","paris_pompidou_000195.jpg",
                 "paris_pompidou_000196.jpg","paris_pompidou_000201.jpg","paris_pompidou_000467.jpg",
                 "paris_pompidou_000640.jpg","paris_sacrecoeur_000299.jpg","paris_sacrecoeur_000330.jpg",
                 "paris_sacrecoeur_000353.jpg","paris_triomphe_000662.jpg","paris_triomphe_000833.jpg",
                 "paris_triomphe_000863.jpg", "paris_triomphe_000867.jpg"]

    with open('lab/all_ok.txt', 'w+') as outfile:
        for fname in all_ok_files:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    with open('lab/all_ok.txt', "r") as f:
        all_ok_file = f.readlines()
    for filename in os.listdir('jpg/1'):
        try:
            im = Image.open("jpg/1/" + filename)
            im.thumbnail((450, 450), Image.ANTIALIAS) # reduce the image
            im.save("jpg/1/" + filename)
            fname = filename[:-4]
            if filename in corrupted:
                os.remove("jpg/1/" + filename)
            elif "general" in fname:
                to_remove=True
                for line in all_ok_file:
                    if fname in line.strip("\n") :
                        to_remove=False
                        break
                if to_remove == True:
                    os.remove("jpg/1/" + filename)
        except:
            pass

""" for each class in the dataset, create file with the names of all the normal images in the class """
def creae_ok_files():
    defense_filenames = ['defense_1_ok.txt', 'defense_2_ok.txt', 'defense_3_ok.txt','defense_4_ok.txt','defense_5_ok.txt']
    with open('lab/defense_ok.txt', 'w+') as outfile:
        for fname in defense_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    eiffel_filenames = [ 'eiffel_1_ok.txt','eiffel_2_ok.txt','eiffel_3_ok.txt','eiffel_4_ok.txt','eiffel_5_ok.txt']
    with open('lab/eiffel_ok.txt', 'w+') as outfile:
        for fname in eiffel_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    invalides_filenames = [ 'invalides_1_ok.txt', 'invalides_2_ok.txt','invalides_3_ok.txt','invalides_4_ok.txt','invalides_5_ok.txt']
    with open('lab/invalides_ok.txt', 'w+') as outfile:
        for fname in invalides_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    louvre_filenames = [ 'louvre_1_ok.txt','louvre_2_ok.txt','louvre_3_ok.txt','louvre_4_ok.txt','louvre_5_ok.txt']
    with open('lab/louvre_ok.txt', 'w+') as outfile:
        for fname in louvre_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    moulinrouge_filenames = ['moulinrouge_1_ok.txt', 'moulinrouge_2_ok.txt','moulinrouge_3_ok.txt','moulinrouge_4_ok.txt','moulinrouge_5_ok.txt']
    with open('lab/moulinrouge_ok.txt', 'w+') as outfile:
        for fname in moulinrouge_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    museedorsay_filenames = [ 'museedorsay_1_ok.txt', 'museedorsay_2_ok.txt', 'museedorsay_3_ok.txt', 'museedorsay_4_ok.txt', 'museedorsay_5_ok.txt' ]
    with open('lab/museedorsay_ok.txt', 'w+') as outfile:
        for fname in museedorsay_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    notredame_filenames = ['notredame_1_ok.txt', 'notredame_2_ok.txt', 'notredame_3_ok.txt', 'notredame_4_ok.txt', 'notredame_5_ok.txt']
    with open('lab/notredame_ok.txt', 'w+') as outfile:
        for fname in notredame_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    pantheon_filenames = ['pantheon_1_ok.txt','pantheon_2_ok.txt','pantheon_3_ok.txt','pantheon_4_ok.txt','pantheon_5_ok.txt']
    with open('lab/pantheon_ok.txt', 'w+') as outfile:
        for fname in pantheon_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    pompidou_filenames = [ 'pompidou_1_ok.txt', 'pompidou_2_ok.txt', 'pompidou_3_ok.txt', 'pompidou_4_ok.txt', 'pompidou_5_ok.txt' ]
    with open('lab/pompidou_ok.txt', 'w+') as outfile:
        for fname in pompidou_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    sacrecoeur_filenames = [ 'sacrecoeur_1_ok.txt','sacrecoeur_2_ok.txt','sacrecoeur_3_ok.txt','sacrecoeur_4_ok.txt','sacrecoeur_5_ok.txt']
    with open('lab/sacrecoeur_ok.txt', 'w+') as outfile:
        for fname in sacrecoeur_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

    triomphe_filenames = ['triomphe_1_ok.txt','triomphe_2_ok.txt','triomphe_3_ok.txt','triomphe_4_ok.txt','triomphe_5_ok.txt']
    with open('lab/triomphe_ok.txt', 'w+') as outfile:
        for fname in triomphe_filenames:
            with open('lab/'+fname) as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == '__main__':
    removing_general()
    creae_ok_files()

